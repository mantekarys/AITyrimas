import airsim
import torch
import numpy as np
from stable_baselines3 import PPO
import time
from collections import deque
import cv2
import random
import numpy as np
import torch
import random
import numpy as np
import torch

import torch.nn.functional as F

def adaptive_instance_normalization(content_feat, style_feat):
    # Compute content and style statistics
    size = content_feat.size()
    content_mean, content_std = calc_mean_std(content_feat)
    style_mean, style_std = calc_mean_std(style_feat)

    # Normalize content features
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)

    # Apply style statistics
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    # Calculate mean and standard deviation
    N, C, H, W = feat.size()
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_camera_image(client, camera_name="0", filename="camera_image.png"):
    # Request image from the camera
    responses = client.simGetImages([
        airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
    ])
    
    if responses and len(responses) > 0:
        response = responses[0]
        # Convert image data to numpy array
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        # Reshape array to image shape
        img_rgb = img1d.reshape(response.height, response.width, 3)
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        # Save the image to disk
        cv2.imwrite(filename, img_bgr)
        print(f"Image saved to {filename}")
    else:
        print("No image received.")


def airsim_env_setup():
    # Connect to AirSim simulator
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = airsim.CarControls()
    return client, car_controls

def load_dino_model():
    # Load the dinov2 model for feature extraction
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
    dinov2 = dinov2.to(device)
    dinov2.eval()  # Ensure the model is in evaluation mode
    return dinov2, device

import torch
import torch.nn.functional as F
import numpy as np

def preprocess_observation(response, dinov2, device, embedding_buffer, style_features):
    """
    Preprocesses the observation from AirSim by applying style transfer to match the style of MetaDrive images.

    Parameters:
    - response: The image response from AirSim.
    - dinov2: The DINOv2 model used for feature extraction.
    - device: The device to run computations on (CPU or CUDA).
    - embedding_buffer: A deque to store embeddings for temporal stacking.
    - style_features: The style features extracted from a MetaDrive image.

    Returns:
    - obs: A dictionary containing the preprocessed 'vit_embeddings' ready for the model.
    """
    # Convert AirSim image response to numpy array
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

    # Ensure that the image data is not empty
    if img1d.size == 0:
        print("Empty image data received.")
        return None  # or handle appropriately

    # Reshape to image shape (assuming RGB image)
    img_rgb = img1d.reshape(response.height, response.width, 3)

    # Convert to a PyTorch tensor and normalize pixel values to [0, 1]
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0  # Shape: [3, H, W]

    # Resize the image to the expected input size for DINOv2 (e.g., 224x224)
    img_resized = F.interpolate(
        img_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
    ).squeeze(0).to(device)  # Shape: [3, 224, 224]

    # Normalize using training data statistics (e.g., ImageNet)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    img_normalized = (img_resized - mean) / std

    # Extract features for content image (AirSim image)
    with torch.no_grad():
        content_features = dinov2.forward_features(img_normalized.unsqueeze(0))['x_norm_patchtokens']  # Shape: [1, num_patches, embedding_dim]

    # Apply style transfer using AdaIN
    stylized_features = adaptive_instance_normalization(content_features, style_features)

    # Remove batch dimension
    stylized_features = stylized_features.squeeze(0)  # Shape: [num_patches, embedding_dim]

    # Append to the embedding buffer
    embedding_buffer.append(stylized_features)

    # If buffer has less than 4 embeddings, pad with zeros
    while len(embedding_buffer) < embedding_buffer.maxlen:
        embedding_buffer.appendleft(torch.zeros_like(stylized_features))

    # Stack embeddings to get shape [4, num_patches, embedding_dim]
    stacked_embeddings = torch.stack(list(embedding_buffer))  # Shape: [4, num_patches, embedding_dim]

    # Convert to numpy and ensure correct data type
    vit_embeddings_np = stacked_embeddings.cpu().numpy().astype(np.float32)

    # Return the expected observation
    return {"vit_embeddings": vit_embeddings_np}

def adaptive_instance_normalization(content_feat, style_feat):
    """
    Performs Adaptive Instance Normalization (AdaIN) on content features using style features.

    Parameters:
    - content_feat: Content features from the content image (AirSim image).
    - style_feat: Style features from the style image (MetaDrive image).

    Returns:
    - Stylized content features.
    """
    # Compute content and style statistics
    content_mean, content_std = calc_mean_std(content_feat)
    style_mean, style_std = calc_mean_std(style_feat)

    # Normalize content features
    normalized_feat = (content_feat - content_mean) / content_std

    # Apply style statistics
    stylized_feat = normalized_feat * style_std + style_mean

    return stylized_feat

def calc_mean_std(feat, eps=1e-5):
    """
    Calculates the mean and standard deviation of the features.

    Parameters:
    - feat: Feature map of shape [1, num_patches, embedding_dim].
    - eps: Small value to avoid division by zero.

    Returns:
    - mean: Mean of the features.
    - std: Standard deviation of the features.
    """
    # Compute mean and standard deviation along the patches dimension
    mean = feat.mean(dim=1, keepdim=True)  # Shape: [1, 1, embedding_dim]
    std = feat.std(dim=1, keepdim=True) + eps  # Shape: [1, 1, embedding_dim]

    return mean, std


def load_style_image(path, device):
    from PIL import Image
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224))
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    # Normalize using the same statistics
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    img_normalized = (img_tensor.to(device) - mean) / std

    return img_normalized


def lane_detection_adjustment(img_rgb):
    # Convert to grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Edge detection
    edges = cv2.Canny(blur, 50, 150)
    # Define a region of interest (ROI)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (width, height * 0.6),
        (0, height * 0.6),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    # Hough Transform to detect lines
    lines = cv2.HoughLinesP(cropped_edges, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    # Calculate steering adjustment
    steering_adjustment = 0.0
    if lines is not None:
        # Compute average slope to adjust steering
        slopes = [(y2 - y1) / (x2 - x1 + 1e-6) for line in lines for x1, y1, x2, y2 in line]
        average_slope = np.mean(slopes)
        steering_adjustment = -average_slope * 0.1  # Adjust factor as needed
    return steering_adjustment


def main(model_path):
    # Set up AirSim environment
    client, car_controls = airsim_env_setup()

    # Load the trained PPO model
    model = PPO.load(model_path)

    # Load the dinov2 feature extractor
    dinov2, device = load_dino_model()

    # Initialize the embedding buffer
    embedding_buffer = deque(maxlen=4)

    previous_acceleration = 0.0
    previous_throttle = 0.0
    previous_brake = 0.0
    previous_steering = 0.0

    save_camera_image(client, camera_name="0", filename="camera_image_before_change.png")

    camera_name = "0"
    # Define the camera pose
    camera_pose = airsim.Pose(
        airsim.Vector3r(x_val=12, y_val=0, z_val=0)  # Position offsets
    )
    # Set the camera pose
    client.simSetCameraPose(camera_name, camera_pose)

    save_camera_image(client, camera_name="0", filename="camera_image_after_change.png")


    # Load the style image
    style_img_tensor = load_style_image('metadrive_style_image.png', device)

    # Extract style features
    with torch.no_grad():
        style_features = dinov2.forward_features(style_img_tensor.unsqueeze(0))['x_norm_patchtokens']  # Shape: [1, num_patches, embedding_dim]

    try:
        while True:
            # Get camera images from the car
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])

            # Ensure response is valid
            if not responses or len(responses) == 0:
                print("No image received.")
                continue

            obs = preprocess_observation(responses[0], dinov2, device, embedding_buffer, style_features)

            if obs is None:
                # Skip if preprocessing failed
                continue

            # Get model action
            action, _states = model.predict(obs, deterministic=True)

            # Ensure action has the correct size
            if len(action) < 2:
                print("Invalid action received.")
                continue

            # Translate the action to car controls with brake
            steering = -1*np.clip(action[0]*0.8, -0.5, 0.5)
            acceleration = np.clip(action[1], -1, 1)

            if acceleration >= 0 and previous_accelaration <= acceleration:
                throttle = np.clip(acceleration, -0.7, 0.7)
                brake = 0.0
            else:
                throttle = 0.0
                brake = np.clip(-acceleration*4,0,1)

            target_speed = 8.0  # Desired speed in m/s
            current_speed = client.getCarState().speed  # Get current speed
            speed_error = target_speed - current_speed
            throttle_adjustment = np.clip(speed_error * 0.1, -1, 1)  # Proportional control factor

            # Adjust throttle based on speed error
            throttle += throttle_adjustment
            throttle = np.clip(throttle, 0.0, 1.0)


            # steering_adjustment = lane_detection_adjustment(img_rgb)
            # steering += steering_adjustment



            previous_accelaration = acceleration

            car_controls.steering = float(steering)
            car_controls.throttle = float(throttle)
            car_controls.brake = float(brake)

            # Send control command to the car
            client.setCarControls(car_controls)

            # Wait before the next step
            time.sleep(0.1)
    except KeyboardInterrupt:
        # Disable API control when the script is stopped
        client.enableApiControl(False)
        print("Control disabled.")
    except Exception as e:
        print(f"An error occurred: {e}")
        client.enableApiControl(False)
        print("Control disabled due to an error.")

if __name__ == "__main__":
    set_seeds(42)
    main("car/chill-owl-867-1")
