"""
Put all Apple and Tomato on the floor
"""


class SceneInitializer:
    def __init__(self) -> None:
        pass

    def preinit(self,event, controller):
        """Pre-initialize the environment for the task.
        Args:
            event: env.event object
            controller: ai2thor.controller object
        Returns:
            event: env.event object
        """
        # set the apple position on the floor
        apple_position = {'x': 0.6107811331748962, 'y': 0, 'z': 0.2376347929239273}
        # set the tomato position on the floor
        tomato_position = {'x':-0.49381884932518005, 'y': 0, 'z': -2.0611681938171387}

        # Teleport the apple and tomato to the given positions on the floor
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Tomato|+00.92|+01.91|+01.61',
        position={'x': -0.8478860259056091, 'y': 0.2598906457424164, 'z': 1.2558666467666626}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Apple|-01.75|+01.37|-01.16',
        position={'x': 0.13871456682682037, 'y': 0.5, 'z': 0.23763470351696014}
        )
        return event
    
    # event=controller.step(
    #     action='PlaceObjectAtPoint',
    #     objectId='Apple|-01.75|+01.37|-01.16',
    #     position={'x': 0.6107811331748962, 'y': 0.2798214256763458, 'z': 0.2376347929239273}
    #     )
        
    #     event=controller.step(
    #     action='PlaceObjectAtPoint',
    #     objectId='Tomato|+00.92|+01.91|+01.61',
    #     position={'x': -0.49381884932518005, 'y': 0.2598906457424164, 'z': -2.0611681938171387}
    #     )