"""
Pre-initalization for FloorPlan1 task.
FloorPlan1 does not need any modifications for the task
of Put all shakers next to the tomato
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
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='Tomato|-00.39|+01.14|-00.81',
        position={'x': -0.2606813609600067, 'y': 0.03703154996037483, 'z': 1.3859623670578003}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='SaltShaker|+00.35|+00.90|-02.57',
        position={'x': 2.0667929649353027, 'y': 0.025804493576288223, 'z': 0.731961727142334}
        )

        return event
