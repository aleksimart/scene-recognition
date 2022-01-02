package ukuk.ac.soton.ecs.grp;


import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.evaluation.classification.BasicClassificationResult;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.*;

public class Run2_2 {


    public static void main (String[] args){
        //List<float[]> patches = getPatches(App.randomInstanceTest, 4, 8);
        //System.out.println(patches.get(0).length);
        //System.out.println(App.splits.getTrainingDataset().numInstances());
        ArrayList<float[]> patches = new ArrayList<>();
        for (FImage image: GroupedUniformRandomisedSampler.sample(App.splits.getTrainingDataset(),30)){
            patches.addAll(getPatches(image, 4, 8));
        }
        HardAssigner<float[], float[], IntFloatPair> assigner = KMeansClustering(patches.toArray(new float[patches.size()][]));
        BOVWExtractor extractor = new BOVWExtractor(assigner);
        LiblinearAnnotator<FImage, String> annotator = new LiblinearAnnotator<FImage, String>(
               extractor , LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001 );
        annotator.train(App.splits.getTrainingDataset() );
        try {
            PrintWriter printWriter = new PrintWriter(new File("run2.txt"));
            for (int i = 0; i < App.testingData.size(); i++) {
                printWriter.println(App.testingData.getID(i).substring(8) + " " + annotator.classify(App.testingData.get(i)).getPredictedClasses().toArray()[0].toString());
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        /*
        ClassificationEvaluator<CMResult<String>, String, FImage> eval =
                new ClassificationEvaluator<CMResult<String>, String, FImage>(
                        annotator, App.splits.getTestDataset(), new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));
        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        Collection<ClassificationResult<String>> classResults = guesses.values();
        for (ClassificationResult<String> classResult :classResults){
            System.out.println(classResult.getPredictedClasses().toString());
        }
        CMResult<String> result = eval.analyse(guesses);
        System.out.println(result);
        */
    }

    private static List<float[]> getPatches(FImage inputImage, int interval, int patchsize){
        RectangleSampler sampler = new RectangleSampler(inputImage, interval, interval, patchsize, patchsize);
        List<Rectangle> rectangles = sampler.allRectangles();

        ArrayList<float[]> vectors = new ArrayList<>();

        for (Rectangle rectangle: rectangles){
            FImage imagePatch = inputImage.extractROI(rectangle);
            /*imagePatch = imagePatch.subtract(Run1.averageFloat(imagePatch));
            imagePatch = imagePatch.normalise();*/
            vectors.add(imagePatch.getFloatPixelVector());
        }
        //float[][] arr = new float[vectors.size()][];
        return vectors;
    }

    private static HardAssigner<float[], float[], IntFloatPair> KMeansClustering(float[][] floatPatches){
        final FloatKMeans kMeans =  FloatKMeans.createExact(500);
        FloatCentroidsResult result = kMeans.cluster(floatPatches);
        HardAssigner<float[], float[], IntFloatPair> hardAssigner =  result.defaultHardAssigner();
        return hardAssigner;
    }

    static class BOVWExtractor implements FeatureExtractor<SparseIntFV, FImage> {
        HardAssigner<float[], float[], IntFloatPair> assigner;

        BOVWExtractor(HardAssigner<float[], float[], IntFloatPair> assigner){
            this.assigner = assigner;
        }

        @Override
        public SparseIntFV extractFeature(FImage object) {
            List<float[]> imageFeatureVectors = getPatches(object, 4,8);
            BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>( assigner );
            return bovw.aggregateVectorsRaw( imageFeatureVectors );
        }

    }
}
