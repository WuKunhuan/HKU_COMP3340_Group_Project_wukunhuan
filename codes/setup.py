
import os, shutil, sys, subprocess

class setup:
    
    def __init__(self): 
        
        os.system("export PATH=/usr/bin:$PATH")
        os.system("export PATH=/bin:$PATH")
        
        os.system("clear")
        if (os.name == 'nt'):
            raise Exception (f"[Error] Please use Linux or Mac, but not Windows to work with this notebook")

        self.path = str(subprocess.check_output("pwd"))
        self.path = self.path[2:len(self.path)-3:1]
        print ("Start setting up ... (this may take some time to download packages)")
        print(f"\nPath: {self.path}")
       

        package_name = ["pip", "pytorch", "numpy", "torchvision", "scikit-learn", "seaborn", "pandas", "zip", "unzip", "wget"]
        package_download = ["conda install ", "conda install ", "pip install ", "pip install ", "pip install ", "pip install ", "pip install ", "conda install ", "conda install ", "pip install "]
        import_name = [None, "torch", "numpy as np", "torchvision", "sklearn", "seaborn as sn", "pandas as pd", None, None, None]
        
        for item in range (len(package_name)): 
            try:
                if (import_name[item] == None): 
                    os.system (f"{package_download[item]}{package_name[item]} > /dev/null")
                else: 
                    exec(f"import {import_name[item]}")
            except Exception:
                try: 
                    os.system (f"{package_download[item]}{package_name[item]} > /dev/null")
                    print (f"[Package] Download {package_name[item]} package successfully. ")
                    exec(f"import {import_name[item]}")
                except Exception: 
                    print (f"[Error] Failed to import the {package_name[item]} package. ")
                    os.system("clear")
                    raise Exception(f"[Error] import {package_name[item]} failed. Run \"{package_download[item]}{package_name[item]}\" in a terminal under {self.path}, then run the setup again. ")
            if (import_name[item] == None): 
                print (f"[Package] install {package_name[item]} successfully. ")
            else: 
                print (f"[Package] import {package_name[item]} successfully. ")
        
        
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
    
