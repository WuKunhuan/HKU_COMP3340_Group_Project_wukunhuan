
import os

try: 
    os.system ("pip install unzip")
    print ("[Package] Download unzip package successfully. ")
except Exception: print ("[Error] Failed to download the sklearn package. ")
import unzip

class Download_Dataset: 
    
    def __init__ (self, setup, link, name): 
        
        self.name = name
        self.link = link
        self.path = setup.path + f"/datasets/{self.name}"
        
        print ("Start downloading the dataset ... ", end = "")
        os.system (f"wget {self.link}")
        
        n = ""
        i = -5
        while (self.link[i] != '/'): 
            n = self.link[i] + n
            i -= 1
        print ("Finished")
            
        print (f"Dataset link: {self.link}")
        print (f"Dataset name: {n}")
        print (f"Proposed name: {self.name}")
        
        os.system (f"unzip {n}")
        os.system (f"mv {n} datasets/{self.name}")
        os.system (f"rm -rf {n}.zip")
        
        os.system (f"unzip {self.link}")
        
        print (f"Finished downloading the {n} dataset. ")
        