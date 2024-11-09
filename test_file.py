from app.core.system import AutoMLSystem

automl = AutoMLSystem.get_instance()

print(automl._registry)