# Get AA and classified sequences composition as well as group transition

def get_aa_compo(sequence: str):
    '''Get the amino acid composition from a sequence
    
    :param sequence: The amino acid sequence (only composed of the 20 essential amino acids)
    :type sequence: str 

    :return: The dictionary with the 20 amino acid composition value (from 0 to 1) of the sequence
    :rtype: dict
    '''
    res = {
            'A': 0, 
            'R': 0, 
            'N': 0, 
            'D': 0, 
            'C': 0, 
            'Q': 0, 
            'E': 0, 
            'G': 0, 
            'H': 0, 
            'I': 0, 
            'L': 0, 
            'K': 0, 
            'M': 0, 
            'F': 0, 
            'P': 0, 
            'S': 0, 
            'T': 0, 
            'W': 0, 
            'Y': 0, 
            'V': 0, 
            'X': 0
        }
    
    for aa in sequence:
        res[aa] += 1/len(sequence)
    
    return res



def get_group_compo(
        sequence: str, 
        classification_mode: str
        ):
    '''Get the composition of amino acid groups from the given classified sequence
    
    :param sequence: The sequence containing group letters (depending of the classification mode)
    :type sequence: str 

    :param classification_mode: The classification mode telling which groups to look for in the 
    sequence. Must be "charac_1", "charac_2" or "charac_3"
    :type classification_mode: str 

    :return: The dictionary containing all the group composition of the given sequence based 
    on the given classification method
    :rtype: dict
    '''

    if classification_mode == "charac_1":
        group_list = ['A', 'B', 'C', 'D']
    elif classification_mode == "charac_2":
        group_list = ['A', 'B', 'C', 'D', 'E']
    elif classification_mode == "charac_3":
        group_list = ['A', 'B', 'C', 'D', 'P', 'G']
    
    res = {}
    for group in group_list:
        grp_name = "grp_"+group
        res[grp_name] = 0
    
    for aa_g in sequence:
        grp_name = "grp_"+aa_g
        res[grp_name] += 1/len(sequence)
    
    return res



def aa_classification_seq(
        sequence: str, 
        classification_mode: str
        ):
    '''Create a sequence of letters corresponding to groups based on the choosen classification 
    from the given sequence
    
    :param sequence: The amino acid sequence to convert 
    (must only contains the 20 essential amino acids)
    :type sequence: str 

    :param classification_mode: The classification mode used for the convertion of an amino acid by 
    its classification group. Must be "charac_1", "charac_2" or "charac_3"    
    :type classification_mode: str  

    :return: The sequence in which all amino acids have been replaced by their corresponding group 
    letter
    :rtype: str
    '''

    if classification_mode == "charac_1":
        groupA = ['R', 'H', 'K', 'D', 'E'] # Charged (positives and negatives)
        groupB = ['S', 'T', 'N', 'Q'] # Polar Uncharged
        groupC = ['C', 'G', 'P'] # Special cases
        groupD = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'] # Hydrophobic
        groupE = [] # empty
    elif classification_mode == "charac_2":
        groupA = ['R', 'H', 'K'] # Charged Positives
        groupB = ['D', 'E'] # Charged Negatives
        groupC = ['S', 'T', 'N', 'Q'] # Polar Uncharged
        groupD = ['C', 'G', 'P'] # Special cases
        groupE = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'] # Hydrophobic
    elif classification_mode == "charac_3":
        groupA = ['H', 'T', 'C', 'S'] # polar
        groupB = ['K', 'R', 'E', 'D'] # charged
        groupC = ['I', 'L', 'M', 'V', 'W', 'Y', 'F', 'A'] # hydrophobic
        groupD = ['Q', 'N'] # 'Important' polar
        groupE = [] # empty
        groupP = ['P']
        groupG = ['G']
    convert_seq = ""

    for aa in sequence:
        if aa in groupA:
            convert_seq += 'A'
        elif aa in groupB:
            convert_seq += 'B'
        elif aa in groupC:
            convert_seq += 'C'
        elif aa in groupD:
            convert_seq += 'D'
        elif aa in groupE:
            convert_seq += 'E'
        elif aa in groupP:
            convert_seq += 'P'
        elif aa in groupG:
            convert_seq += 'G'
        else:
            print(f'No aa classification for: {aa}')
            return None
    return convert_seq



def get_group_transition(
        sequence: str, 
        classification_mode: str
        ):
    '''Get the transition proportion from a group ID to an other (from N to C-term)
    
    :param sequence: The sequence containing group letters (depending on the choosen 
    classification method)
    :type sequence: str 

    :param classification_mode: The classification mode telling which groups to look for in the 
    sequence. Must be "charac_1", "charac_2" or "charac_3"
    :type classification_mode: str 

    :return: The dictionary containing, for all possible group transition, their composition in the 
    sequence (from 0 to 1)
    :rtype: dict
    '''
    
    if classification_mode == "charac_1":
        group_list = ['A', 'B', 'C', 'D']
    elif classification_mode == "charac_2":
        group_list = ['A', 'B', 'C', 'D', 'E']
    elif classification_mode == "charac_3":
        group_list = ['A', 'B', 'C', 'D', 'P', 'G']
    
    res = {}

    for grp1 in group_list:
        for grp2 in group_list:
            transi = grp1 + '_to_' + grp2
            res[transi] = 0
    
    for i in range(len(sequence)-1):
        transi_name = sequence[i] + '_to_' + sequence[i+1]
        res[transi_name] += 1/len(sequence)
    
    return res
