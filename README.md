### Codebase for paper titled "Enhancing Categorization of Computer Science Research Papers using Knowledge Bases" from KG4IR workshop at SIGIR'17

The paper presents a method to enhance categorization of computer science research papers into one of 24 pre-defined classes. (For ex. 'Programming Language', 'Machine Learning and Pattern Recognition', 'Computer Vision' etc.)

Standard text classification methods work by converting text into a feature representation (tf-idf, distributed representation etc.) and through a parameterized function, generate probability distribution over all classes. A loss function is defined then between the parameterized probability distribution function and the actual class distribution of the data-point (multinomial distribution). Then a minimizer is employed which adjusts the parameters to minimize the loss.

In this kind of setup, no semantic knowledge of the classes is used. For example, in the dataset considered, classes like - 'Machine Learning and Pattern Recognition', 'Computer Vision' have lot of semantic information available (ex. their wikipedia article).

In this work, integration of such semantic knowledge of the classes is attempted with the text classification pipeline. The problem is formulated as a Learning-To-Rank (LTR) problem where the goal is to match a scientific article to it's corresponding category.

Considering an analogy to std. LTR use-case (query-document matching, with a query matched to only one document) it works as follows:



* The query in this case is the scientific article's abstract. The document against which the query is to be mapped is the category of the scientific article ('Data mining', 'Programming language' etc).

* Knowledge-Base consists of entities and their relations. Each category is mapped to it's corresponding entity in KB by matching their corresponding description text. For ex. the category 'Machine Learning and Pattern Recognition' is mapped to the entity '/en/machine_learning' in Freebase and to entity 'Machine learning' in Wikipedia.

* After mapping category to the corresponding entity in the KB, feature representation for both abstract and category (entity) are generated. In this work, doc2vec embedding for article, doc2vec embedding for entity (for Wikipedia KB), relational embedding for entity (for Freebase KB).

* A pointwise LTR model is trained to match the article with it's category.

#### Dataset Link:

The dataset used in the paper, i.e the scientific articles can be found from the following link:

https://drive.google.com/file/d/0B-7peEFiNjnUZk5DV3RzZlZveGM/view?usp=sharing


#### Text Classification Category to KB entity mapping

##### Mappings for Freebase

'Programming languages' = '/en/programming_language'
'Real time and embedded systems' = '/en/embedded_system'
'Scientific computing' = '/en/scientific_computing'
'Natural language and speech' = '/en/natural_language_processing'
'Machine learning and pattern recognition' = '/en/machine_learning'
'Operating systems' = '/en/operating_system'
'World wide web' = '/en/world_wide_web'
'Bioinformatics and computational biology' = '/en/bioinformatics'
'Security and privacy'= '/en/internet_security'
'Distributed and parallel computing' = '/en/parallel_computing'
'Databases' = '/en/database'
'Simulation' = '/en/simulation'
'Algorithms and theory' = '/en/algorithm'
'Computer education' = '/en/computer_literacy'
'Human-computer interaction'= '/en/human_computer_interaction'
'Hardware and architecture'= '/en/hardware_architecture'
'Networks and communications' = '/en/computer_network'
'Artificial intelligence' = '/en/artificial_intelligence'
'Data mining' = '/en/data_mining'
'Computer vision' = '/en/computer_vision'
'Simulation' = '/en/simulation'
'Software engineering' = '/en/software_engineering'
'Information retrieval'= '/en/information_retrieval'
'Multimedia' = '/en/multimedia'
'Graphics' ='/en/graphics'

##### Mappings for Wikipedia
'Programming languages' = 'Programming language'; ['Real time and embedded systems'] = ['Modeling and Analysis of Real Time and Embedded systems'];
['Scientific computing'] = ['Computational science'];
['Natural language and speech'] = ['Natural language processing','Speech recognition'];
['Machine learning and pattern recognition'] = ['Machine learning','Pattern recognition']; ['Operating systems'] = ['Operating system'];
['World wide web'] = ['World Wide Web'];  ['Bioinformatics and computational biology'] = ['Bioinformatics','Computational biology'];
['Security and privacy']=['Information security', 'Internet privacy']; mapFields['Distributed and parallel computing'] = ['Distributed computing','Parallel computing'];
['Databases'] = ['Database'];
['Simulation'] = ['Computer simulation'];
['Algorithms and theory'] = ['Algorithm', 'Theoretical computer science'];
['Computer education']=['Computer literacy'];
['Human-computer interaction']= [];
['Hardware and architecture']=['Hardware architecture'];
['Networks and communications'] = ['Computer network', 'Telecommunications engineering']
['Artificial intelligence'] = ['Artificial intelligence']; mapFields['Data mining'] = ['Data mining'];
['Computer vision'] = ['Computer vision']; ['Simulation'] = ['Simulation'] ; ['Software engineering'] = ['Software engineering'];
['Information retrieval']=['Information retrieval']; mapFields['Multimedia'] = ['Multimedia']; mapFields['Graphics'] = ['Graphics']



#### Instructions:

* Download the [wikipedia data dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2]) in xml format and put it in the current folder.

* Train doc2vec by running the file createDV.py. It will generate and save doc2vec embeddings of all wikipedia articles indexed by its title.

* run baseline* files to reproduce experiments. For ex. to reproduce tf-idf baseline experiment, run baseline_if-idf.py
