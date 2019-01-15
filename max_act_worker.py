import numpy as np

"""
NOTE: This script is only necessary, because the max_activation.txt files were not readable before. (See maximum_pathway.py)
Change of those is necessary!
"""


# Compare the top 9 activations per layer to find out whether a channel is specific for one organ, family, genus or specie

def get_class_name_from_xml(file):
  """Read a XML file and returns the information 
  Args:
    file: path to the xml file
      
  Returns:
     class_id: Id of Plant 
     family: Family of plant
     species: Species of plant
     genus: Genus of plant
     media_id: number of picture/file
  """
  tree = ET.parse(file)
  root = tree.getroot()
  class_id = int(tree.find("ClassId").text) # int(root[5].text)
  family = tree.find("Family").text # root[6].text
  species = tree.find("Species").text #root[7].text
  genus =  tree.find("Genus").text#root[8].text
  media_id = tree.find("MediaId").text# root[2].text
  organ = tree.find("Content").text
    
  return class_id, family, species, genus, media_id, organ


def img_to_xml(filename):
    media_id = filename.split("'")[1]
    media_id = media_id.split(".")[0]
    return ("mydata/train/%s.xml" % (media_id))

def make_dict_file(dictionary, filename):
  """Reads the labels file and returns a mapping from ID to class name.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    A map from a label (integer) to class name.
  """
  with open(filename, "a") as f:
      for k, v in dictionary.items():
          f.write("%s:" % k)
          f.write("%s" % v[0])
          for item in v[1:]:
              f.write(",%s" % item)
          f.write("\n")   
          
def read_dict_file(filename):
    with open(filename, "r") as f:
        lines = f.read()
    lines = lines.split("\n")
    #lines = filter(None, lines)


    return_dict = {}
    for line in lines[:-1]:
        index = line.index(':')
        list_of_lists = []
        its = line[index+1:].split(",")
        for i in range(0, len(its), 3):
            channel = int(its[i+1])
            activation = float(its[i])
            image = its[i+2]   
            list_of_lists.append([activation, channel, image])
            
        
        
        return_dict[line[:index]] = list_of_lists
    return return_dict   


def from_tf_file_to_readale():
    """
    Transorms the max-act dicts to readable dict files
    """

    max_act_v0 = my_read_label_file("","max_activations_v0.txt")
    max_act_v1 = my_read_label_file("","max_activations_v1.txt")



    for k, v in max_act_v0.items():
        max_act_v0[k] = {}
    
        v = v.replace("{", "")
        v = v.replace("}", "")   
            
        v = v.replace("[", "")
        v = v.replace("]", "")
        
    
        
        ind = v.index(":",1)
        v = v[ind:]
        v = v.replace(":", "")
        
        
        v = v.split(",")
        
        v[0] = " " + v[0]
        
        v_0 =[]
        for index, v_i in enumerate(v):
            
            if index % 27 == 0:
                v_0.append(v_i.split(" ")[2])
            else:   
                v_0.append(v_i)
        max_act_v0[k] = v_0 
    make_dict_file(max_act_v0, "max_act_v0_readable.txt")
            
            
    for k, v in max_act_v1.items():
        max_act_v1[k] = {}
    
        v = v.replace("{", "")
        v = v.replace("}", "")   
            
        v = v.replace("[", "")
        v = v.replace("]", "")
        
    
        
        ind = v.index(":",1)
        v = v[ind:]
        v = v.replace(":", "")
        
        
        v = v.split(",")
        
        v[0] = " " + v[0]
        
        v_0 =[]
        for index, v_i in enumerate(v):
            
            if index % 27 == 0:
                v_0.append(v_i.split(" ")[2])
            else:   
                v_0.append(v_i)
        max_act_v1[k] = v_0  
        
        
    make_dict_file(max_act_v0, "max_act_v1_readable.txt")    
    
######################

from_tf_file_to_readale()
    

max_act_v0 = read_dict_file("max_act_v0_readable.txt")
max_act_v1 = read_dict_file("max_act_v1_readable.txt")


for key, values in max_act_v0.items():
    for i in range(0, len(values), 9):
        channel_list = values[i:i+9]
        
        no_families, no_species, no_genus, no_organs = 1, 1, 1, 1
        channel = channel_list[0][1]
        _, default_family, default_species, default_genus, _, default_organ = get_class_name_from_xml(img_to_xml(channel_list[0][2]))
        
        for sublist in channel_list[1:]: # [activation, channel, img]
            _, family, species, genus, _, organ = get_class_name_from_xml(img_to_xml(sublist[2]))
            no_families += (family == default_family)
            no_species += (species == default_species)
            no_genus += (genus == default_genus)
            no_organs += (organ == default_organ)
        if no_families == 9:
            with open("activation_statistics_v0.txt", "a") as f:
                f.write("Channel %s in Layer %s recognized family %s" % (channel, key, default_family))
        if no_genus == 9:
            with open("activation_statistics_v0.txt", "a") as f:
                f.write("Channel %s in Layer %s recognized genus %s" % (channel, key, default_genus))
        if no_species == 9:	
            with open("activation_statistics_v0.txt", "a") as f:
                f.write("Channel %s in Layer %s recognized species %s" % (channel, key, default_species))
        if no_organs == 9:
            with open("activation_statistics_v0.txt", "a") as f:
                f.write("Channel %s in Layer %s recognized organ %s" % (channel, key, default_organ))

for key, values in max_act_v1.items():
    for i in range(0, len(values), 9):
        channel_list = values[i:i+9]
        
        no_families, no_species, no_genus, no_organs = 1, 1, 1, 1
        channel = channel_list[0][1]
        _, default_family, default_species, default_genus, _, default_organ = get_class_name_from_xml(img_to_xml(sublist[2]))
        
        for sublist in channel_list[1:]: # [activation, channel, img]
            _, family, species, genus, _, organ = get_class_name_from_xml(img_to_xml(channel_list[0][2]))
            no_families += (family == default_family)
            no_species += (species == default_species)
            no_genus += (genus == default_genus)
            no_organs += (organ == default_organ)
        if no_families == 9:
            with open("activation_statistics_v0.txt", "a") as f:
                f.write("Channel %s in Layer %s recognized family %s" % (channel, key, default_family))
        if no_genus == 9:
            with open("activation_statistics_v0.txt", "a") as f:
                f.write("Channel %s in Layer %s recognized genus %s" % (channel, key, default_genus))
        if no_species == 9:	
            with open("activation_statistics_v0.txt", "a") as f:
                f.write("Channel %s in Layer %s recognized species %s" % (channel, key, default_species))
        if no_organs == 9:
            with open("activation_statistics_v0.txt", "a") as f:
                f.write("Channel %s in Layer %s recognized organ %s" % (channel, key, default_organ))






















































