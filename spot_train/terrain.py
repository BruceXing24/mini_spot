import pybullet_data as pd
import pybullet as p
import random

class Terrain:
    def __init__(self, terrain_name, columns=256, rows=256):
        random.seed(random.randint(0,30))
        self.terrain_source = terrain_name
        self.columns = columns
        self.rows = rows

    def generate_terrain(self,pybullet_client, height_perturbation_range=0.05):
        # pybullet_client.setAdditionalSearchPath(pd.getDataPath())
        # pybullet_client.configureDebugVisualizer(pybullet_client.COV_ENABLE_RENDERING, 0)
        # height_perturbation_range = height_perturbation_range
        terrain_data = [0] * self.columns * self.rows
        if self.terrain_source == 'random':
            for j in range(int(self.columns / 2)):
                for i in range(int(self.rows / 2)):
                    height = random.uniform(0, height_perturbation_range)
                    terrain_data[2 * i + 2 * j * self.rows] = height
                    terrain_data[2 * i + 1 + 2 * j * self.rows] = height
                    terrain_data[2 * i + (2 * j + 1) * self.rows] = height
                    terrain_data[2 * i + 1 + (2 * j + 1) * self.rows] = height
            terrain_shape = pybullet_client.createCollisionShape(
                shapeType=pybullet_client.GEOM_HEIGHTFIELD,
                meshScale=[.025, .025, 1],
                heightfieldTextureScaling=(self.rows - 1) / 2,
                heightfieldData=terrain_data,
                numHeightfieldRows=self.rows,
                numHeightfieldColumns=self.columns)

            terrain = pybullet_client.createMultiBody(0, terrain_shape)
            pybullet_client.changeVisualShape(terrain, -1, rgbaColor=[0.5, 0.5, 0, 0.5])
            pybullet_client.loadTexture("texture/grass.png")
            return terrain

        else:
            return
            # pybullet_client.changeVisualShape(terrain, -1, rgbaColor=[255, 0, 0, 0.5])
            # pybullet_client.loadTexture("texture/grass.png")
            # pybullet_client.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1])
