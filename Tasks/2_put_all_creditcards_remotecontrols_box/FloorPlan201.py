"""
Pre-initalization for FloorPlan201 task.
FloorPlan201 does not need any modifications for the task of putting all the credit cards and remote controls in the box
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
        # Box|-03.36|+00.19|+06.43
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Box|-03.36|+00.19|+06.43',
        position={'x': -4.5, 'y': 0.19336192309856415, 'z': 2.20195821762085}
        )
        print(event.metadata["errorMessage"])
        return event
