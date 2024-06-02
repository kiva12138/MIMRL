from types import SimpleNamespace

from Config import dataset_dimensions
from DataLoaderAVEC2019 import *
from DataLoaderCMUSDK import *
from DataLoaderCMUDeclareLab import *
from DataLoaderLocal import *


def get_data_loader(opt):
    dataset = opt.dataset
    text, audio, video=opt.text, opt.audio, opt.video # Only for CMUSDK dataset and avec
    normalize, log_scale = opt.normalize, opt.log_scale,
    time_len = opt.time_len
    persistent_workers=opt.persistent_workers
    batch_size, num_workers, pin_memory, drop_last =opt.batch_size, opt.num_workers, opt.pin_memory, opt.drop_last

    assert dataset in ['mosi_SDK', 'mosei_SDK', 'pom_SDK', 'mosi_20', 'mosi_50', 'youtube', 'youtubev2', 'mmmo', 'mmmov2', 'moud', 'pom', 'iemocap_20', 'mosei_20', 'mosei_50', 'avec2019', 'mosi_Dec', 'mosei_Dec']

    if 'SDK' in dataset:
        if 'mosi' in dataset:
            dataset_train = CMUSDKDataset(mode='train', dataset='mosi', text=text, audio=audio, video=video, normalize=normalize, log_scale=log_scale, )
            dataset_valid = CMUSDKDataset(mode='valid', dataset='mosi', text=text, audio=audio, video=video, normalize=normalize, log_scale=log_scale, )
            dataset_test = CMUSDKDataset(mode='test', dataset='mosi', text=text, audio=audio, video=video, normalize=normalize, log_scale=log_scale, )
            data_loader_train = DataLoader(dataset_train, batch_size, collate_fn=multi_collate_mosei_mosi, shuffle=True, 
                persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
            data_loader_valid = DataLoader(dataset_valid, batch_size, collate_fn=multi_collate_mosei_mosi, shuffle=False, 
                persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
            data_loader_test = DataLoader(dataset_test, batch_size, collate_fn=multi_collate_mosei_mosi, shuffle=False, 
                persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
            return data_loader_train, data_loader_valid, data_loader_test, dataset_dimensions['mosi_SDK'][0][text], dataset_dimensions['mosi_SDK'][1][audio], dataset_dimensions['mosi_SDK'][2][video]
        
        if 'mosei' in dataset:
            dataset_train = CMUSDKDataset(mode='train', dataset='mosei', text=text, audio=audio, video=video, normalize=normalize, log_scale=log_scale, )
            dataset_valid = CMUSDKDataset(mode='valid', dataset='mosei', text=text, audio=audio, video=video, normalize=normalize, log_scale=log_scale, )
            dataset_test = CMUSDKDataset(mode='test', dataset='mosei', text=text, audio=audio, video=video, normalize=normalize, log_scale=log_scale, )
            data_loader_train = DataLoader(dataset_train, batch_size, collate_fn=multi_collate_mosei_mosi, shuffle=True, 
                persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
            data_loader_valid = DataLoader(dataset_valid, batch_size, collate_fn=multi_collate_mosei_mosi, shuffle=False, 
                persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
            data_loader_test = DataLoader(dataset_test, batch_size, collate_fn=multi_collate_mosei_mosi, shuffle=False, 
                persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
            return data_loader_train, data_loader_valid, data_loader_test, dataset_dimensions['mosei_SDK'][0][text], dataset_dimensions['mosei_SDK'][1][audio], dataset_dimensions['mosei_SDK'][2][video]

        if 'pom' in dataset:
            dataset_train = CMUSDKDataset(mode='train', dataset='pom', text=text, audio=audio, video=video, normalize=normalize, log_scale=log_scale, )
            dataset_valid = CMUSDKDataset(mode='valid', dataset='pom', text=text, audio=audio, video=video, normalize=normalize, log_scale=log_scale, )
            dataset_test = CMUSDKDataset(mode='test', dataset='pom', text=text, audio=audio, video=video, normalize=normalize, log_scale=log_scale, )
            data_loader_train = DataLoader(dataset_train, batch_size, collate_fn=multi_collate_pom, shuffle=True, 
                persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
            data_loader_valid = DataLoader(dataset_valid, batch_size, collate_fn=multi_collate_pom, shuffle=False, 
                persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
            data_loader_test = DataLoader(dataset_test, batch_size, collate_fn=multi_collate_pom, shuffle=False, 
                persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
            return data_loader_train, data_loader_valid, data_loader_test, dataset_dimensions['pom_SDK'][0][text], dataset_dimensions['pom_SDK'][1][audio], dataset_dimensions['pom_SDK'][2][video]
    elif 'Dec' in dataset:
        data_loader_train = get_loader(dataset_name=dataset, mode='train', batch_size=batch_size, time_len=time_len, shuffle=True, 
            persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
        data_loader_valid = get_loader(dataset_name=dataset, mode='valid', batch_size=batch_size, time_len=time_len, shuffle=False, 
            persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
        data_loader_test = get_loader(dataset_name=dataset, mode='test', batch_size=batch_size, time_len=time_len, shuffle=False, 
            persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
            
        if 'mosi' in dataset:
            return data_loader_train, data_loader_valid, data_loader_test, dataset_dimensions['mosi_dec'][0], dataset_dimensions['mosi_dec'][1], dataset_dimensions['mosi_dec'][2]
        elif 'mosei' in dataset:
            return data_loader_train, data_loader_valid, data_loader_test, dataset_dimensions['mosei_dec'][0], dataset_dimensions['mosei_dec'][1], dataset_dimensions['mosei_dec'][2]
        else:
            raise NotImplementedError
    elif 'avec2019' in dataset:
        dataset_train = AVEC2019Dataset(mode='train', dataset='avec2019', text=text, audio=audio, video=video, normalize=normalize, log_scale=log_scale, )
        dataset_valid = AVEC2019Dataset(mode='valid', dataset='avec2019', text=text, audio=audio, video=video, normalize=normalize, log_scale=log_scale, )
        dataset_test = AVEC2019Dataset(mode='test', dataset='avec2019', text=text, audio=audio, video=video, normalize=normalize, log_scale=log_scale, )
        data_loader_train = DataLoader(dataset_train, batch_size, collate_fn=multi_collate_avec, shuffle=True, 
            persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
        data_loader_valid = DataLoader(dataset_valid, batch_size, collate_fn=multi_collate_avec, shuffle=False, 
            persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        data_loader_test = DataLoader(dataset_test, batch_size, collate_fn=multi_collate_avec, shuffle=False, 
            persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        return data_loader_train, data_loader_valid, data_loader_test, dataset_dimensions['avec2019'][0][text], dataset_dimensions['avec2019'][1][audio], dataset_dimensions['avec2019'][2][video]
    else:
        dataset_train = LocalDataset(mode='train', dataset=dataset, normalize=normalize, log_scale=log_scale, )
        dataset_valid = LocalDataset(mode='valid', dataset=dataset, normalize=normalize, log_scale=log_scale, )
        dataset_test = LocalDataset(mode='test', dataset=dataset, normalize=normalize, log_scale=log_scale, )
        data_loader_train = DataLoader(dataset_train, batch_size, shuffle=True, 
                persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
        data_loader_valid = DataLoader(dataset_valid, batch_size, shuffle=False, 
                persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        data_loader_test = DataLoader(dataset_test, batch_size, shuffle=False, 
                persistent_workers=persistent_workers, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

        return data_loader_train, data_loader_valid, data_loader_test, dataset_dimensions[dataset][0], dataset_dimensions[dataset][1], dataset_dimensions[dataset][2]


    raise NotImplementedError


def get_dataset_scales():
    mins = {'mosi_SDK':[1000,1000,1000], 'mosei_SDK':[1000,1000,1000], 'pom_SDK':[1000,1000,1000], 'mosi_20':[1000,1000,1000], 'mosi_50':[1000,1000,1000], 'mosei_20':[1000,1000,1000], 'mosei_50':[1000,1000,1000], 'youtube':[1000,1000,1000], 'youtubev2':[1000,1000,1000], 'mmmo':[1000,1000,1000], 'mmmov2':[1000,1000,1000], 'moud':[1000,1000,1000], 'pom':[1000,1000,1000], 'iemocap_20':[1000,1000,1000], 'mosi_Dec':[1000,1000,1000], 'mosei_Dec':[1000,1000,1000]}
    maxs = {'mosi_SDK':[-1000,-1000,-1000], 'mosei_SDK':[-1000,-1000,-1000], 'pom_SDK':[-1000,-1000,-1000], 'mosi_20':[-1000,-1000,-1000], 'mosi_50':[-1000,-1000,-1000], 'mosei_20':[-1000,-1000,-1000], 'mosei_50':[-1000,-1000,-1000], 'youtube':[-1000,-1000,-1000], 'youtubev2':[-1000,-1000,-1000], 'mmmo':[-1000,-1000,-1000], 'mmmov2':[-1000,-1000,-1000], 'moud':[-1000,-1000,-1000], 'pom':[-1000,-1000,-1000], 'iemocap_20':[-1000,-1000,-1000], 'mosi_Dec':[-1000,-1000,-1000], 'mosei_Dec':[-1000,-1000,-1000]}
    for dataset_name in ['mosi_Dec', 'mosei_Dec', 'mosi_SDK', 'mosei_SDK', 'pom_SDK', 'mosi_20', 'mosi_50', 'mosei_20', 'mosei_50', 'youtube', 'youtubev2', 'mmmo', 'mmmov2', 'moud', 'pom', 'iemocap_20']:
        print('='*40, dataset_name, '='*40,)
        text_min, audio_min, video_min = 1000, 1000, 1000
        text_max, audio_max, video_max = -1000, -1000, -1000
        args = SimpleNamespace(dataset=dataset_name,
            text='glove', audio='covarep', video='facet42', time_len=200, 
            normalize=[False, False, False], log_scale=[False, False, False],
            batch_size=10240, persistent_workers=False, num_workers=0, pin_memory=False, drop_last=False)
        data_loader_train, data_loader_valid, data_loader_test, _, _, _ = get_data_loader(args)
        for i, data in enumerate(data_loader_train):
            text, audio, video = data[0], data[1], data[2]
            text_min, audio_min, video_min = min(text.min().item(), text_min), min(audio.min().item(), audio_min), min(video.min().item(), video_min)
            text_max, audio_max, video_max = max(text.max().item(), text_max), max(audio.max().item(), audio_max), max(video.max().item(), video_max)
        for i, data in enumerate(data_loader_valid):
            text, audio, video = data[0], data[1], data[2]
            text_min, audio_min, video_min = min(text.min().item(), text_min), min(audio.min().item(), audio_min), min(video.min().item(), video_min)
            text_max, audio_max, video_max = max(text.max().item(), text_max), max(audio.max().item(), audio_max), max(video.max().item(), video_max)
        for i, data in enumerate(data_loader_test):
            text, audio, video = data[0], data[1], data[2]
            text_min, audio_min, video_min = min(text.min().item(), text_min), min(audio.min().item(), audio_min), min(video.min().item(), video_min)
            text_max, audio_max, video_max = max(text.max().item(), text_max), max(audio.max().item(), audio_max), max(video.max().item(), video_max)
        mins[dataset_name] = [text_min, audio_min, video_min]
        maxs[dataset_name] = [text_max, audio_max, video_max]
        print(mins)
        print(maxs)


def test_one_dataset():
    args = SimpleNamespace(dataset='mosi_SDK', text='glove', audio='covarep', video='facet41',
        normalize=[False, True, True], log_scale=[False, True, True],
        batch_size=10240, persistent_workers=False, num_workers=0, pin_memory=False, drop_last=False)
    data_loader_train, data_loader_valid, data_loader_test, d_t, d_a, d_v = get_data_loader(args)
    for i, data in enumerate(data_loader_train):
        text, audio, video = data[0], data[1], data[2]
        print(text.shape, audio.shape, video.shape)
        assert text.shape[-1]==d_t and audio.shape[-1]==d_a and video.shape[-1]==d_v, 'Error in '+str([d_t, d_a, d_v])


def test_all_dataset():
    for dataset_name in ['mosi_Dec', 'mosei_Dec', 'mosi_SDK', 'mosei_SDK', 'pom_SDK', 'mosi_20', 'mosi_50', 'mosei_20', 'mosei_50', 'youtube', 'youtubev2', 'mmmo', 'mmmov2', 'moud', 'pom', 'iemocap_20', 'avec2019']:
        print('='*40, dataset_name, '='*40,)
        args = SimpleNamespace(dataset=dataset_name,
            text='glove', audio='covarep' if dataset_name != 'avec2019' else 'ds', video='facet42' if dataset_name != 'avec2019' else 'resnet',
            normalize=[False, True, True], log_scale=[False, True, True], time_len=100,
            batch_size=10240, persistent_workers=False, num_workers=0, pin_memory=False, drop_last=False)
        print(args)
        data_loader_train, data_loader_valid, data_loader_test, d_t, d_a, d_v = get_data_loader(args)
        for i, data in enumerate(data_loader_train):
            text, audio, video = data[0], data[1], data[2]
            print(text.shape, audio.shape, video.shape)
            # assert text.shape[-1]==d_t and audio.shape[-1]==d_a and video.shape[-1]==d_v, 'Error in '+str([d_t, d_a, d_v])
            assert audio.shape[-1]==d_a and video.shape[-1]==d_v, 'Error in '+str([d_t, d_a, d_v])


if __name__ == '__main__':
    # get_dataset_scales()
    test_all_dataset()
