from preprocessing import *

def make_dataset(path_to_dataset):
    dataset = []
    for root, dirs, files in os.walk(path_to_dataset):
        print(dirs)
        for file in files:
            if file.endswith('.wav'):
                filename = os.path.join(root, file)
                samples, samplingrate = loadAudio(filename)
                lmfcc, mspec = mfcc(samples, return_mspec=True)

                # Phoneme is just the description the phoneme for the utterance, (start, end), not the target
                phonemes = Get_phonemes(filename.replace('.wav', '.phn'))

                target = Make_target(samples, phonemes)
                sentence = Get_words(filename.replace('.wav', '.txt'))

                dataset.append(
                    {'filename': filename, 'lmfcc': lmfcc, 'mspec': mspec, 'phonemes': phonemes, 'sentence': sentence,
                     'target': target})

    dataset_name = path_to_dataset.split('/')[-1]
    np.savez(dataset_name + 'data.npz', data=dataset)