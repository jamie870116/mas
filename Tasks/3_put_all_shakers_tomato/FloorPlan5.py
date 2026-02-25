"""
Pre-initialization for FloorPlan4 task.
FloorPlan5 does not need any modifications for the task
of Put all shakers next to the tomato
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
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='PepperShaker|+00.71|+00.90|-01.82',
        position={'x': -0.3481927514076233, 'y': 8.464977145195007e-05, 'z': 1.5051791667938232}
        )
        
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='SaltShaker|+01.19|+00.90|-02.18',
        position={'x': 1.1917768716812134, 'y': 8.406862616539001e-05, 'z': -1.2252197265625}
        )

        return event
