# HypAR: Hypergraph with Attention on Reviews for Explainable Recommendation

This is the code for the paper:

**[Hypergraph with Attention on Reviews for Explainable Recommendation](https://www.ecir2024.org/accepted-paper/)**
<br>
[Theis E. Jendal](https://vbn.aau.dk/da/persons/141501), [Trung-Hoang Le](http://lthoang.com/), 
[Hady W. Lauw](http://www.hadylauw.com/), [Matteo Lissandrini](https://people.cs.aau.dk/~matteo/),
[Peter Dolog](https://peterdolog.wordpress.com), and [Katja Hose](https://homes.cs.aau.dk/~khose/About_me.html)
<br>
Presented at [ECIR 2024](https://www.ecir2024.org/).

## How to run
Install requirements. To use cuda, use 'requirements_cu116txt' instead.

```bash
pip install -r requirements.txt
```

HypAR uses https://github.com/evison/Sentires to preprocess the data. 
The processed data can be downloaded from: [https://drive.google.com/drive/folders/1DZAUJ4aO5O4SLxCFGS3MYBQhn9DlZ7qE](https://github.com/PreferredAI/seer.

After downloading file, simply run
    
```bash
python hypar_example.py
```

## Contanct
Questions and discussion are welcome: [tjendal@cs.aau.dk](mailto:tjendal@cs.aau.dk)
