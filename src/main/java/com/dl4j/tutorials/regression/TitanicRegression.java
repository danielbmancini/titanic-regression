package com.dl4j.tutorials.regression;

// Importações das bibliotecas necessárias para pré-processamento de dados, configuração da rede neural, treinamento e avaliação.
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

// Classe principal para treinar um modelo de regressão nos dados do Titanic.
public class TitanicRegression {

    // Configuração do logger para saída de depuração.
    private static final Logger log = LoggerFactory.getLogger(TitanicRegression.class);

    public static void main(String[] args) throws Exception {
        // Configuração básica do treinamento.
        final int batchSize = 500; // Número de exemplos processados por vez.
        final int nEpochs = 50;    // Número de épocas de treinamento.
        int seed = 31415;          // Semente para inicialização aleatória.
        double learningRate = 0.1; // Taxa de aprendizado.
        int numInputs = 7;         // Número de entradas da rede neural.
        int numOutputs = 1;        // Número de saídas (para regressão, normalmente é 1).

        // Divisão dos dados em treinamento e teste.
        SplitTestAndTrain testAndTrain = getSplitTestAndTrain(batchSize);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        // Configuração e construção da rede neural.
        MultiLayerConfiguration conf = buildMultiLayerConfiguration(seed, learningRate, numInputs, numOutputs);
        MultiLayerNetwork net = buildMultiLayerNetwork(conf);

        // Treinamento da rede neural.
        log.debug("Fit training data");
        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainingData);
        }

        // Avaliação do modelo nos dados de teste.
        RegressionEvaluation eval = new RegressionEvaluation(1);
        INDArray output = net.output(testData.getFeatures(), false);
        eval.eval(testData.getLabels(), output);

        // Impressão das métricas de avaliação.
        log.debug("MSE: " + eval.averageMeanSquaredError());
        log.debug("MAE: " + eval.averageMeanAbsoluteError());
        log.debug("RMSE: " + eval.averagerootMeanSquaredError());
        log.debug("RSE: " + eval.averagerelativeSquaredError());
        log.debug("Correlation: " + eval.averagePearsonCorrelation());
        log.debug("R^2: " + eval.averageRSquared());
    }

    // Configuração do modelo de rede neural.
    private static MultiLayerNetwork buildMultiLayerNetwork(MultiLayerConfiguration conf) {
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1)); // Exibe a pontuação do modelo após cada iteração.
        return net;
    }

    private static MultiLayerConfiguration buildMultiLayerConfiguration(int seed, double learningRate, int numInputs, int numOutputs) {
        // Configuração da arquitetura da rede neural.
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123) // Semente fixa para resultados reproduzíveis.
                .updater(new Adam(0.01)) // Otimizador Adam para treinamento.
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(numInputs) // Número de neurônios de entrada.
                        .nOut(10)       // Número de neurônios na camada oculta.
                        .activation(Activation.RELU) // Função de ativação ReLU.
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE) // Camada de saída com função de perda MSE.
                        .nIn(10)
                        .nOut(numOutputs) // Número de neurônios de saída.
                        .activation(Activation.IDENTITY) // Função de ativação linear para regressão.
                        .build())
                .build();

        return conf;
    }

    // Divisão dos dados em conjunto de treinamento e teste.
    private static SplitTestAndTrain getSplitTestAndTrain(int batchSize) throws IOException, InterruptedException {
        CSVRecordReader csvRecordReader = new CSVRecordReader(1, ',');
        FileSplit inputSplit = new FileSplit(new File("src/main/resources/processed_train.data"));
        csvRecordReader.initialize(inputSplit);

        Schema schema = buildSchema(); // Define o esquema dos dados.
        TransformProcess transformProcess = buildTransformProcess(schema); // Processo de transformação dos dados.
        Schema finalSchema = transformProcess.getFinalSchema();

        TransformProcessRecordReader trainRecordReader = new TransformProcessRecordReader(csvRecordReader, transformProcess);
        RecordReaderDataSetIterator trainIterator = new RecordReaderDataSetIterator.Builder(trainRecordReader, batchSize)
                .regression(finalSchema.getIndexOfColumn("Survived")) // Configuração para regressão.
                .build();

        DataSet allData = trainIterator.next();
        normalizeDataSet(allData); // Normaliza os dados.
        allData.shuffle(123); // Embaralha os dados.

        return allData.splitTestAndTrain(0.8); // Divide 80% para treinamento e 20% para teste.
    }

    private static void normalizeDataSet(DataSet allData) {
        // Normaliza os dados para média 0 e desvio padrão 1.
        NormalizerStandardize normalizerStandardize = new NormalizerStandardize();
        normalizerStandardize.fit(allData);
        normalizerStandardize.transform(allData);
    }

    private static TransformProcess buildTransformProcess(Schema schema) {
        // Aplica transformações nos dados, como remoção de colunas e codificação one-hot.
        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .removeColumns("PassengerId") // Remove colunas desnecessárias.
                .integerToOneHot("Sex", 0, 1) // Codifica a coluna "Sex" em one-hot.
                .build();
        return transformProcess;
    }

    private static Schema buildSchema() {
        // Define o esquema dos dados de entrada.
        Schema schema = new Schema.Builder()
                .addColumnInteger("PassengerId")
                .addColumnInteger("Survived")
                .addColumnInteger("Pclass")
                .addColumnInteger("Sex")
                .addColumnInteger("Age")
                .addColumnInteger("SibSp")
                .addColumnsInteger("Parch")
                .addColumnsInteger("Fare")
                .build();
        return schema;
    }
}
