from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sparknlp.annotator import *
from sparknlp.base import *

paisaCorpusPath = "C:\\Users\\Nabil\\PycharmProjects\\pipelineTesi\\paisa.raw.utf8"

file_path = 'C:\\Users\\Nabil\\PycharmProjects\\pipelineTesi\\nomi.txt'


def get_pipeline():
    pipeline = Pipeline(
        stages=[
            assembler,
            tokenizer,
            loaded
        ])
    fitPipeline = pipeline.fit(df)
    return LightPipeline(fitPipeline)


spark = SparkSession.builder \
    .appName("Spark NLP") \
    .master("local[*]") \
    .config("spark.driver.memory", "16G") \
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.kryoserializer.buffer.max", "2000M") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.0.2") \
    .getOrCreate()

# do some brief DS exploration, and preparation to get clean text
df = spark.read.text(paisaCorpusPath)
df = df.filter(~col('value').contains('</text')). \
    filter(~col('value').contains('<text')). \
    filter(~col('value').startswith('#'))  # .\
# limit(10000)
df.limit(2).show()

with open(file_path, 'r') as file:
    lines = file.readlines()
    names = [name.strip() for line in lines for name in line.split(',')]

assembler = DocumentAssembler().setInputCol("value").setOutputCol("document")

tokenizer = RecursiveTokenizer() \
    .setInputCols("document") \
    .setOutputCol("token") \
    .setInfixes(["’"]) \
    .setWhitelist(["L’unica"]) \
    .setPrefixes(["\"", "“", "(", "[", "\n", ".", "l’", "dell’", "nell’", "sull’", "all’", "d’", "un’"]) \
    .setSuffixes(["\"", "”", ".", ",", "?", ")", "]", "!", ";", ":"])

# we're going to add a special class for names, and use another two
# that come predefined with the model: numbers and dates
spellChecker = ContextSpellCheckerApproach(). \
    setInputCols("token"). \
    setOutputCol("corrected"). \
    addVocabClass('_NAME_', names). \
    setLanguageModelClasses(1650). \
    setWordMaxDistance(3). \
    setBatchSize(24). \
    setEpochs(8)

loaded = ContextSpellCheckerModel.load('./italian_spell')


def correct_line(text_line):
    lp = get_pipeline()
    result = lp.annotate(text_line)['corrected']
    corrected_text = ' '.join(result)
    return corrected_text
