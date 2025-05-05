from models import UNet

def set_model_params() -> dict:
    """Sets model parameters."""
    model_params = {
        "in_channels": 3,
        "out_channels": 1,
        "epochs": 100, 
    }
    return model_params

def main():
    model_params = set_model_params()

    model = UNet(in_channels=model_params["in_channels"], 
                 out_channels=model_params["out_channels"])   
    
    # Train/test model here

if __name__ == "__main__":
    main()