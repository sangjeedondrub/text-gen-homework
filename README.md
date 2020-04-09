# Text Generation with RNNs

> Homework assignment by prof.Huaque

# Github

<https://github.com/sangjeedondrub/text-gen-homework>


# Environment

    pip install --upgrade tensorflow

    import tensorflow as tf

    print(tf.__version__)

    2.1.0


# Data

    head -n 10 lu-drub-gong-gyan.txt

དབུ་མའི་ཟབ་གནད་སྙིང་པོར་དྲིལ་བའི་ལེགས་བཤད་ཀླུ་སྒྲུབ་དགོངས་རྒྱན་ཞེས་བྱ་བ་བཞུགས་སོ༎
དགེ་འདུན་ཆོས་འཕེལ།
བདུད་ཀྱི་མཚོན་ཆ་རྩུབ་མོ་དག་ལ་འཇམ་མཉེན་མེ་ཏོག་ལན་དུ་སྦྱིན༎
ལྷས་བྱིན་ཁྲོས་པའི་སྒྱོགས་རྡོ་འཕངས་ཚེ་མི་སྨྲའི་བརྟུལ་ཞུགས་དང་དུ་བླངས༎
རང་གི་དགྲ་ལ་སྡང་མིག་ཙམ་ཡང་བལྟ་བར་མི་ནུས་ཤཱཀྱའི་སྲས༎
འཇིགས་རུང་འཁོར་བའི་དགྲ་ཆེན་སྐྱོབས་པའི་གྲོགས་སུ་ཤེས་ལྡན་སུ་ཡིས་བཀུར༎
དགེ་ལེགས་ཞི་བསིལ་བདུད་རྩིའི་འབྱུང་གནས་ཐར་པའི་ལམ༎
འདུན་པས་ཕྲ་ཞིབ་མ་ནོར་མཚོན་པ་སྲིད་པའི་མིག།
ཆོས་རྗེ་སྨྲ་བའི་སེང་གེ་མཆོག་ནི་འདིའོ་ཞེས༎
འཕེལ་འགྲིབ་མེད་པར་རིགས་སྨྲ་ཚོགས་ཀྱིས་རྟག་ཏུ་གུས༎
གང་གི་མཁྱེན་པའི་ཉི་མའི་དཀྱིལ་འཁོར་ཀུན་ཏུ་བཟང་པོའི་མཁའ་དབྱིངས་ནས༎}



Total number of lines

    wc -l lu-drub-gong-gyan.txt

    918 lu-drub-gong-gyan.txt

<table>
<caption class="t-above"><span class="table-number">Table 1:</span> As Character</caption>

<colgroup>
<col  class="org-left">

<col  class="org-right">
</colgroup>
<tbody>
<tr>
<td class="org-left">Total number of characters</td>
<td class="org-right">81674</td>
</tr>


<tr>
<td class="org-left">Total vocab</td>
<td class="org-right">65</td>
</tr>


<tr>
<td class="org-left">Total Patterns</td>
<td class="org-right">81664</td>
</tr>
</tbody>
</table>

<table>
<caption class="t-above"><span class="table-number">Table 2:</span> As Syllable</caption>

<colgroup>
<col  class="org-left">

<col  class="org-right">
</colgroup>
<tbody>
<tr>
<td class="org-left">Total number of syllables</td>
<td class="org-right">21755</td>
</tr>


<tr>
<td class="org-left">Total vocab</td>
<td class="org-right">1189</td>
</tr>


<tr>
<td class="org-left">Totoal pattern</td>
<td class="org-right">21752</td>
</tr>
</tbody>
</table>


# Models

Character model

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    embedding_1 (Embedding)      (64, None, 256)           16640
    _________________________________________________________________
    gru_1 (GRU)                  (64, None, 1024)          3938304
    _________________________________________________________________
    dense_1 (Dense)              (64, None, 65)            66625
    =================================================================
    Total params: 4,021,569
    Trainable params: 4,021,569
    Non-trainable params: 0

Syllable model

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    embedding_4 (Embedding)      (1, None, 256)            304384
    _________________________________________________________________
    gru_4 (GRU)                  (1, None, 1024)           3938304
    _________________________________________________________________
    dense_4 (Dense)              (1, None, 1189)           1218725
    =================================================================
    Total params: 5,461,413
    Trainable params: 5,461,413
    Non-trainable params: 0
    _________________________________________________________________


# Train and evaluate

    git clone https://github.com/sangjeedondrub/text-gen-homework --depth=1
    cd text-gen-homework

Train character-level model

    python text-gen.py

Train and evaluate syllable-level model

    python text-gen.py --use_syllable

The evaluation results will be saved to `sample.syllable.txt` and
`sample.char.txt` files
