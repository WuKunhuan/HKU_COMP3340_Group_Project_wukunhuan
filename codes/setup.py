
import os, shutil, sys, subprocess

class setup:
    
    def __init__(self): 
        os.system("clear")
        if (os.name == 'nt'):
            raise Exception (f"[Error] Please use Linux or Mac, but not Windows to work with this notebook")

        self.path = str(subprocess.check_output("pwd"))
        self.path = self.path[2:len(self.path)-3:1]
        print(f"Path: {self.path}")

        # Import Pytorch
        try:
            import torch
            import torch.nn as nn
        except Exception:
            try: 
                os.system ("pip install pytorch")
                print ("[Package] Download pytorch package successfully. ")
                import torch
                import torch.nn as nn
            except Exception: 
                print ("[Error] Failed to download the pytorch package. ")
                os.system("clear")
                raise Exception(f"[Error] import pytorch failed. Run \"conda install pytorch\" in a terminal under {self.path}, then run the setup again. ")
        print ("[Package] import pytorch succeeded. ")

        # Import numpy
        try:
            import numpy as np
        except Exception:
            try: 
                os.system ("pip install numpy")
                print ("[Package] Download numpy package successfully. ")
                import numpy as np
            except Exception: 
                print ("[Error] Failed to download the numpy package. ")
                os.system("clear")
                raise Exception(f"[Error] import numpy failed. Run \"conda install numpy\" in a terminal under {self.path}, then run the setup again. ")
        print ("[Package] import numpy succeeded. ")
            

        # Import torchvision
        try:
            import torchvision
            from torchvision import transforms
        except Exception:
            try: 
                os.system ("pip install torchvision")
                print ("[Package] Download torchvision package successfully. ")
                import torchvision
                from torchvision import transforms
            except Exception: 
                os.system("clear")
                raise Exception(f"[Error] import torchvision failed. Run \"conda install torchvision\" in a terminal under {self.path}, then run the setup again. ")
        print ("[Package] import torchvision succeeded. ")
        
        if (os.path.exists(self.path + "/figures")): 
            pass
        else: 
            shutil.rmtree(self.path + "/figures", ignore_errors=True)
            os.makedirs(self.path + "/figures", exist_ok = True)
            print ("[Folder] Created the figures folder. ")
        if (os.path.exists(self.path + "/datasets")): 
            pass
        else: 
            shutil.rmtree(self.path + "/datasets", ignore_errors=True)
            os.makedirs(self.path + "/datasets", exist_ok = True)
            print ("[Folder] Created the datasets folder. ")
        if (os.path.exists(self.path + "/trained_models")): 
            pass
        else: 
            shutil.rmtree(self.path + "/trained_models", ignore_errors=True)
            os.makedirs(self.path + "/trained_models", exist_ok = True)
            print ("[Folder] Created the trained_models folder. ")
        
        print ("")

        print ("COMP3340_GP setup finished. ")
    



