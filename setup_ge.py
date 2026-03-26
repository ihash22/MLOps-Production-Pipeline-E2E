import great_expectations as gx
import os

def initialize_ge_v1():
    project_path = "./"
    ge_dir = os.path.join(project_path, "gx") # v1 often defaults to 'gx' instead of 'great_expectations'
    
    try:
        # get_context() will initialize a new FileDataContext if project_root_dir is provided
        # and no context exists there yet.
        context = gx.get_context(project_root_dir=project_path)
        
        print(f"✅ Great Expectations (v1.x) initialized!")
        print(f"Context type: {type(context).__name__}")
        
    except Exception as e:
        print(f"⚠️ Initialization issue: {e}")

if __name__ == "__main__":
    initialize_ge_v1()