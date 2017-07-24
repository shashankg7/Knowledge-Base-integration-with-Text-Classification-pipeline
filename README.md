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


#### Instructions:

* Download the [wikipedia data dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2]) in xml format and put it in the current folder.

* Train doc2vec by running the file createDV.py. It will generate and save doc2vec embeddings of all wikipedia articles indexed by its title.

* run baseline* files to reproduce experiments. For ex. to reproduce tf-idf baseline experiment, run baseline_if-idf.py
