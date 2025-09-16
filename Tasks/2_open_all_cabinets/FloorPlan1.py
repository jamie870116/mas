
"""Close all cabinets(9) for pre-initialization

Cabinet': {'|+00.68|+00.50|-02.20': 1, '|-01.18|+00.50|-02.20': 2, '|-01.55|+00.50|+00.38': 3, '|+00.72|+02.02|-02.46': 4, '|-01.85|+02.02|+00.38': 5, '|+00.68|+02.02|-02.46': 6, '|-01.55|+00.50|-01.97': 7, '|-01.69|+02.02|-02.46': 8, '|-00.73|+02.02|-02.46': 9}
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
        action='CloseObject',
        objectId='Cabinet|+00.68|+00.50|-02.20',
        forceAction=True
        )
        
        event=controller.step(
        action='CloseObject',
        objectId='Cabinet|-01.18|+00.50|-02.20',
        forceAction=True
        )
        
        event=controller.step(
        action='CloseObject',
        objectId='Cabinet|-01.55|+00.50|+00.38',
        forceAction=True
        )
        
        event=controller.step(
        action='CloseObject',
        objectId='Cabinet|+00.72|+02.02|-02.46',
        forceAction=True
        )
        
        event=controller.step(
        action='CloseObject',
        objectId='Cabinet|-01.85|+02.02|+00.38',
        forceAction=True
        )
        
        event=controller.step(
        action='CloseObject',
        objectId='Cabinet|+00.68|+02.02|-02.46',
        forceAction=True
        )
        
        event=controller.step(
        action='CloseObject',
        objectId='Cabinet|-01.55|+00.50|-01.97',
        forceAction=True
        )
        
        event=controller.step(
        action='CloseObject',
        objectId='Cabinet|-01.69|+02.02|-02.46',
        forceAction=True
        )
        
        event=controller.step(
        action='CloseObject',
        objectId='Cabinet|-00.73|+02.02|-02.46',
        forceAction=True
        )
        
        return event
        