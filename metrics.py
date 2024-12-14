def calculate_average_time_and_distance(episodes):
  """
  Calculate the average time and distance traveled by an RL agent.

  Parameters:
  episodes (list of dict): A list of episode dictionaries, each containing 'time' and 'distance' keys.

  Returns:
  tuple: A tuple containing the average time and average distance.
  """
  total_time = 0
  total_distance = 0
  num_episodes = len(episodes)

  for episode in episodes:
    total_time += episode['time']
    total_distance += episode['distance']

  average_time = total_time / num_episodes if num_episodes > 0 else 0
  average_distance = total_distance / num_episodes if num_episodes > 0 else 0

  return average_time, average_distance

# Example usage:
# episodes = [{'time': 10, 'distance': 100}, {'time': 15, 'distance': 150}, {'time': 20, 'distance': 200}]
# avg_time, avg_distance = calculate_average_time_and_distance(episodes)
# print(f"Average Time: {avg_time}, Average Distance: {avg_distance}")
