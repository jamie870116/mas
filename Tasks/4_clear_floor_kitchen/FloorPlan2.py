"""
Put all Apple and Tomato on the floor
"""


class SceneInitializer:
    def __init__(self) -> None:
        pass

    def preinit(self, event, controller):
        """Pre-initialize the environment for the task.
        Args:
            event: env.event object
            controller: ai2thor.controller object
        Returns:
            event: env.event object
        """
        # set the apple position on the floor
        apple_position = {'x': 1.104349136352539, 'y': 0, 'z':  2.7725017070770264}
        # set the tomato position on the floor
        tomato_position = {'x': -0.7016176581382751, 'y': 0, 'z': -0.8815637826919556}

        # Teleport the apple and tomato to the given positions on the floor
        event = controller.step(
            action="PlaceObjectAtPoint",
            objectId='Apple|-01.65|+00.81|+00.07',
            position=apple_position,
        )
        event = controller.step(
            action="PlaceObjectAtPoint",
            objectId='Tomato|+00.17|+00.97|-00.28',
            position=tomato_position,
        )

        return event
    
    # event=controller.step(
    #     action='PlaceObjectAtPoint',
    #     objectId='Apple|-01.65|+00.81|+00.07',
    #     position={'x': 1.104349136352539, 'y': 0.05266478657722473, 'z': 2.7725017070770264}
    #     )
        
    #     event=controller.step(
    #     action='PlaceObjectAtPoint',
    #     objectId='Tomato|+00.17|+00.97|-00.28',
    #     position={'x': -0.7016176581382751, 'y': 0.05950151011347771, 'z': -0.8815637826919556}
    #     )