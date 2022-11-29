
import os

class Download_Dataset: 
    
    def __init__ (self, setup, link, name): 
        
        self.name = name
        self.link = link
        self.path = setup.path + f"/datasets/{self.name}"
        
        print ("Start downloading the dataset (it may take some time) ... ", end = "")
        os.system (f"wget -q {self.link} > /dev/null")
        
        n = ""
        i = -5
        while (self.link[i] != '/'): 
            n = self.link[i] + n
            i -= 1
        print ("Finished")
            
        print (f"Dataset link: {self.link}")
        print (f"Dataset name: {n}")
        print (f"Proposed name: {self.name}")
        
        print ("\nStart unzipping the dataset (it may take some time) ... ", end = "")
        os.system (f"unzip {n} > /dev/null")
        os.system (f"mv {n} datasets/{self.name}")
        os.system (f"rm -rf {n}.zip")
        print ("Finished")
        
        print (f"Finished downloading the {n} dataset. ")
        