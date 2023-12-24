import numpy as np
import random
import matplotlib.pyplot as plt


def make_syllables():
    """Make all 336 CV syllables by combining the 24 consonants and 14 vowels"""
    consonants = [
        "B",
        "CH",
        "D",
        "DH",
        "F",
        "G",
        "HH",
        "JH",
        "K",
        "L",
        "M",
        "N",
        "NG",
        "P",
        "R",
        "S",
        "SH",
        "T",
        "TH",
        "V",
        "W",
        "Y",
        "Z",
        "ZH"
    ]
    vowels = [
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "EH",
        "EY",
        "IH",
        "IY",
        "OW",
        "OY",
        "UH",
        "UW"
    ]
    syllables = []
    for c in consonants:
        for v in vowels:
            syllable = f"{c}.{v}"
            syllables.append(syllable)
    return syllables


def make_words(sylls, nwords=1000):
    """Given the set of all possible syllables and the number of words, randomly generate this many words
    in a Zipfian distribution"""
    words = []
    frequencies = []
    # sample word lengths based on a Poisson distribution with mean 2, and add 1
    word_lengths = np.random.poisson(lam=2, size=nwords) + 1
    for i in range(len(word_lengths)):
        # make the word by appending that many syllables
        word = ""
        for j in range(word_lengths[i]):
            word += f"{random.choice(sylls)}|"
        words.append(word[:-1])
        # make the Zipfian distribution
        frequencies.append(nwords / (i + 1))
    return words, frequencies


def make_sentences(words, frequencies, total_tokens=60000):
    """Given a set of words and their frequencies, generate sentences up to a certain number
    of tokens"""
    tokens_generated = 0
    sentences = []
    # keep generating until we reach the total number of tokens we want to generate
    while tokens_generated < total_tokens:
        # sample sentence length from Poisson distribution with mean 2 and add 2
        sentence_length = (np.random.poisson(lam=2, size=1) + 2)[0]
        tokens_generated += sentence_length
        sentence = ""
        curr_word = ""
        while sentence_length > 0:
            # generate the next word randomly and make sure it doesn't equal the last word
            next_word = random.choices(words, weights=frequencies)[0]
            while next_word == curr_word:
                next_word = random.choices(words, weights=frequencies)[0]
            curr_word = next_word
            sentence += f"{next_word} "
            sentence_length = sentence_length - 1
        # add the sentence to the list of generated sentences
        sentences.append(sentence[:-1])
    return sentences


def zipfian_check(sentences, save_path=None):
    """Sanity check that the words are still in a Zipfian frequency distribution once they're
    in the sentences"""
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(8, 4)
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    # get the frequency counts for each word in the input
    word_counts = {}
    for sentence in sentences:
        for word in sentence.split():
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1
    # plot the raw counts and ranks
    counts = np.asarray(sorted(word_counts.values(), reverse=True))
    ranks = np.asarray(range(len(counts)))
    ax1.plot(ranks, counts)
    # plot the log-scaled counts and ranks
    log_counts = np.log(counts)
    log_ranks = np.log(ranks)
    ax2.plot(log_ranks, log_counts)
    # general plotting stuff
    ax1.set_xlabel("Raw Rank")
    ax1.set_ylabel("Raw Frequency")
    ax2.set_xlabel("Log Rank")
    ax2.set_ylabel("Log Frequency")
    plt.suptitle("Zipfian Frequency Distribution of Words in the Stimulus")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    syllables = make_syllables()
    words, freqs = make_words(syllables)
    print(words, freqs)
    sentences = make_sentences(words, freqs, 60000)
    print(sentences)
    zipfian_check(sentences, "test.png")
