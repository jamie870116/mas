"""
Pre-initalization for FloorPlan2 task.
FloorPlan3 needs to move the tomato out of the fridge
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
        objectId='Tomato|+00.92|+01.91|+01.61',
        position={'x': -1.437359094619751, 'y': 2.4899539589881896, 'z': 0.906745195388794}
        )
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='SaltShaker|+00.33|+01.31|-02.83',
        position={'x': -0.8521996736526489, 'y': 0.22221410274505615, 'z': 0.3164134621620178}
        )
        
        event=controller.step(
        action='PlaceObjectAtPoint',
        objectId='PepperShaker|+00.47|+01.31|-03.00',
        position={'x': 0.46660885214805603, 'y': 0.22221243381500244, 'z': 0.13981083035469055}
        )

        return event
