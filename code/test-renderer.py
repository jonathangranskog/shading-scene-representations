from renderer.interface import RenderInterface

# Script that can visualize existing scenes without using any neural network interface
# Simply change the scene in RenderInterface to compare another scene

if __name__ == "__main__":
    interface = RenderInterface(128, -1, scene='room', hidden=False)
