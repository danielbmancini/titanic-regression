package com.dl4j.tutorials.regression;

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
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class TitanicRegression {

    private static final Logger log = LoggerFactory.getLogger(TitanicRegression.class);

    public static void main(String[] args) throws Exception {
        final int batchSize = 500;
        final int nEpochs = 50;
        int seed = 31415;
        double learningRate = 0.1;
        int numInputs = 8;
        int numOutputs = 1;

        SplitTestAndTrain testAndTrain = getSplitTestAndTrain(batchSize);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        MultiLayerConfiguration conf = buildMultiLayerConfiguration(seed, learningRate, numInputs, numOutputs);
        MultiLayerNetwork net = buildMultiLayerNetwork(conf);

        log.debug("Fit training data");
        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainingData);
        }

        RegressionEvaluation eval = new RegressionEvaluation(1);
        INDArray output = net.output(testData.getFeatures(), false);
        eval.eval(testData.getLabels(), output);
        log.debug("" + eval.averageMeanSquaredError());
        log.debug("" + eval.averageMeanAbsoluteError());
        log.debug("" + eval.averagerootMeanSquaredError());
        log.debug("" + eval.averagerelativeSquaredError());
        log.debug("" + eval.averagePearsonCorrelation());
        log.debug("" + eval.averageRSquared());
    }

    private static MultiLayerNetwork buildMultiLayerNetwork(MultiLayerConfiguration conf) {
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));
        return net;
    }

    private static MultiLayerConfiguration buildMultiLayerConfiguration(int seed, double learningRate, int numInputs, int numOutputs) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.01))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(7)  // Change this to 6 to match the dataset
                        .nOut(10)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(10)
                        .nOut(1)  // For regression or adjust for classification
                        .activation(Activation.IDENTITY)
                        .build())
                .build();

        return conf;
    }


    private static SplitTestAndTrain getSplitTestAndTrain(int batchSize) throws IOException, InterruptedException {


        CSVRecordReader csvRecordReader = new CSVRecordReader(1, ',');
        FileSplit inputSplit = new FileSplit(new File("src/main/resources/processed_train.data"));
        csvRecordReader.initialize(inputSplit);


        Schema schema = buildSchema();
        TransformProcess transformProcess = buildTransformProcess(schema);
        Schema finalSchema = transformProcess.getFinalSchema();

        TransformProcessRecordReader trainRecordReader = new TransformProcessRecordReader(csvRecordReader, transformProcess);
        RecordReaderDataSetIterator trainIterator = new RecordReaderDataSetIterator.Builder(trainRecordReader, batchSize)
                .regression(finalSchema.getIndexOfColumn("Survived"))
                .build();

        DataSet allData = trainIterator.next();
        normalizeDataSet(allData);
        allData.shuffle(123);

        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.8);  //Use 65% of data for training
        return testAndTrain;
    }

    private static void normalizeDataSet(DataSet allData) {
        NormalizerStandardize normalizerStandardize = new NormalizerStandardize();
        normalizerStandardize.fit(allData);
        normalizerStandardize.transform(allData);
    }

    private static TransformProcess buildTransformProcess(Schema schema) {
        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .removeColumns("PassengerId")
               // .removeColumns("Parch")
               // .stringToCategorical("Sex", Arrays.asList("m","f"))
                .integerToOneHot("Sex",0,1)
             //   .integerToOneHot("Survived",0,1)
                .build();
        return transformProcess;
    }

    private static Schema buildSchema() {
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
