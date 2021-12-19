# Unsupervised text recognition
This is an implementation of the paper **Learning to Read by Spelling**  by Gupta, Vedaldi, Zisserman (2018).
This paper develops an **unsupervised learning** method for **text recognition**, 
based on convolutional neural networks and adversarial training against real sample of text data.

# Generate datasets only : 

- Generate lexicons from raw text (one for image dataset generation,
one for positives examples in adversarial training) 
```
  $ python3 generate_lexicon.py 
```

- Generate image dataset from first lexicon
```
  $ python3 generate_dataset.py 
```

# Generate datasets and train the model :
```
  $ python3 main.py 
```

Arguments : 
 - -lp : Lexicon path (write and read) default : "data/lexicons/translation_dataset/"
 - -ip : Image dataset path (write and read) default : "data/imgs/translation_dataset/"
 - -sl : String length (write and read) default : 30
 - -il : Image dataset length default : 50000
 - -gl : Generate lexicon from raw text ? default : False
 - -gi : Generate image dataset from lexicon ? default : False

