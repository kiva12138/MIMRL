# Data_path_local = r'E:\MMDataset'
# Data_path_SDK = r'E:\CMU_SDK_Dataset'
# Data_path_DecLab = r'E:\CMU_DeclareLab_Dataset'
Data_path_local = r'/newdata/sh/MMDatasets/Dataset'
Data_path_SDK = r'/newdata/sh/MMDatasets/CMU_SDK_Dataset'
Data_path_DecLab = r'/newdata/sh/MMDatasets/CMU_DeclareLab_Dataset'
# CUDA = "0"
CUDA = "0, 1"
# CUDA = "0, 1, 2"

######## Dataset Scales #########
# Text, Audio, Video
dataset_scales_mins = {
    'mosi_SDK':[{'glove':-4.209499835968018, 'last_hidden_state':-10.058603286743164, 'masked_last_hidden_state':-5.3045454025268555, 'summed_last_four_states': -74.48263549804688}, {"covarep":-33.80808639526367, "opensmile_eb10":-146.04649353027344, 'opensmile_is09':-129.2928924560547}, {"facet41":-25.375, "facet42":-34.783599853515625, "openface": -273.6381530761719}],
    'mosei_SDK':[{'glove':-4.144499778747559, 'last_hidden_state':-9.9786052703857424, 'masked_last_hidden_state':-5.502565860748291, 'summed_last_four_states': -82.04771423339844}, {"covarep":-55.55973434448242}, {"facet42":-39.54077911376953}],
    'pom_SDK':[{'glove':-3.9363999366760254, 'last_hidden_state':-9.604168891906738, 'masked_last_hidden_state':-3.9924752712249756, 'summed_last_four_states': -77.31072998046875}, {"covarep":-515.626708984375}, {"facet42":-33.53656005859375}],
    'avec2019':[{}, {'mfcc':-2.8860552310943604, 'ege':-5.093098163604736, 'ds':0}, {'au':-25.711212158203125, "resnet":-0.062034472823143005}],

    'mosi_dec': [None, -3.141394853591919, -1.3032554388046265], 
    'mosei_dec': [None, -60.02680587768555, -39.62459945678711], 

    'mosi_20': [-4.209499835968018, -3.1244829037090844, -1.1956999464146485], 
    'mosi_50': [-4.209499835968018, -3.127065511312953, -1.1956999464146485], 
    'mosei_20': [-4.1445, -53.92286823758954, -74.92696535587311], 
    'mosei_50': [-3.0639, -55.55973434448242, -39.54077911376953], 
    'youtube': [-4.2095, -24.91410728225707, -3.7277956008911133], 
    'youtubev2': [-4.2095, -24.91410728225707, -25.397525310516357], 
    'mmmo': [-3.9364, -255.0, -5.5620880126953125], 
    'mmmov2': [-3.9364, -255.0, -31.60834422111511], 
    'moud': [-0.285044, -25.564596279948393, -26.948311686515808], 
    'pom': [-3.5302, -255.0, -24.64705433862077], 
    'iemocap_20': [-4.2095, -38.03445016707095, -24.715965747833252]
}

dataset_scales_maxs = {
    'mosi_SDK':[{'glove':3.960900068283081, 'last_hidden_state':4.4694366455078125, 'masked_last_hidden_state':1.5511236190795898, 'summed_last_four_states': 17.987661361694336}, {"covarep":477.75, "opensmile_eb10":34292.94921875, 'opensmile_is09':43021.3046875}, {"facet41":680.5, "facet42":24.77629852294922, "openface": 843.5923461914062}],
    'mosei_SDK':[{'glove':4.190100193023682, 'last_hidden_state':5.137068271636963, 'masked_last_hidden_state':1.7913602590560913, 'summed_last_four_states': 34.28007507324219}, {"covarep":500.0}, {"facet42":30.693572998046875}],
    'pom_SDK':[{'glove':3.960900068283081, 'last_hidden_state':5.215933322906494, 'masked_last_hidden_state':1.1109845638275146, 'summed_last_four_states': 32.44478225708008}, {"covarep":9146.2919921875}, {"facet42":28.543701171875}],
    'avec2019':[{'mfcc':5.680467128753662, 'ege':17.769824981689453, 'ds':12.069750785827637, 'au':20.82292366027832, "resnet":22.89879608154297}],

    'mosi_dec': [None, 3.1415038108825684, 1.4117268323898315], 
    'mosei_dec': [None, 500.0, 31.594900131225586], 

    'mosi_20': [3.960900068283081, 3.132475224615101, 1.6667884934594241], 
    'mosi_50': [3.960900068283081, 3.132475224615101, 1.6667884934594241], 
    'mosei_20': [4.1901, 499.38722666199953, 55.152244210243225], 
    'mosei_50': [2.6668, 500.0, 29.55523109436035], 
    'youtube': [3.9609, 450.9189131105791, 6.210602283477783], 
    'youtubev2': [3.9609, 450.9189131105791, 20.756671726703644], 
    'mmmo': [3.9609, 255.0, 3.8879380226135254], 
    'mmmov2': [3.9609, 255.0, 20.168131828308105], 
    'moud': [0.264721, 464.6709277242704, 23.913885951042175], 
    'pom': [3.7333, 255.0, 255.0], 
    'iemocap_20': [3.9609, 498.55227272731696, 25.14185881614685]
}

dataset_dimensions = {
    'mosi_SDK':[
        {'glove':300, 'last_hidden_state':768, 'masked_last_hidden_state':768, 'summed_last_four_states': 768, 'text': 768}, 
        {"covarep":74, "opensmile_eb10":1585, 'opensmile_is09':384}, 
        {"facet41":47, "facet42":35, "openface": 430}],
    'mosei_SDK':[
        {'glove':300, 'last_hidden_state':768, 'masked_last_hidden_state':768, 'summed_last_four_states': 768, 'text': 768}, 
        {"covarep":74}, 
        {"facet42":35}],
    'pom_SDK':[
        {'glove':300, 'last_hidden_state':768, 'masked_last_hidden_state':768, 'summed_last_four_states': 768, 'text': 768},  
        {"covarep":43}, 
        {"facet42":35}],
    'avec2019':[
        {'text': 768}, 
        {'mfcc':39, 'ege':23, 'ds':1920}, 
        {'au':49, "resnet":2048}],
        
    'mosi_dec': [768, 5, 20],
    'mosei_dec': [768, 74, 35, ],

    'mosi_20': [300, 5, 20],
    'mosi_50': [300, 5, 20],
    'mosei_20': [300, 74, 35],
    'mosei_50': [300, 74, 35],
    'youtube': [300, 74, 36],
    'youtubev2': [300, 74, 35],
    'mmmo': [300, 74, 36],
    'mmmov2': [300, 74, 35],
    'moud': [300, 74, 35],
    'pom': [300, 43, 43],
    'iemocap_20': [300, 74, 35],
}
