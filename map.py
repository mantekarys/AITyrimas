from metadrive import MetaDriveEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.component.map.pg_map import MapGenerateMethod
import matplotlib.pyplot as plt
from metadrive import MetaDriveEnv
from metadrive.utils.draw_top_down_map import draw_top_down_map
import logging

map_config={BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE, 
            BaseMap.GENERATE_CONFIG: "XOS",  # 3 block
            BaseMap.LANE_WIDTH: 3.5,
            BaseMap.LANE_NUM: 2}

fig, axs = plt.subplots(1, 1, figsize=(10, 10), dpi=200)

map_config["config"]="SSSTSSSTSSSTSSSTSSSTSSST"
env = MetaDriveEnv(dict(num_scenarios=10, map_config=map_config, log_level=logging.WARNING))

env.reset(seed=0)

m = draw_top_down_map(env.current_map)
print(axs)
# ax = axs[0][0]
axs.imshow(m, cmap="bone")
axs.set_xticks([])
axs.set_yticks([])
env.close()
plt.show()
