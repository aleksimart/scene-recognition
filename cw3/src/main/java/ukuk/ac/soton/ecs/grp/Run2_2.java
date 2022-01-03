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

        /*
        code used for testing purposes
         */
        //List<float[]> patches = getPatches(App.randomInstanceTest, 4, 8);
        //System.out.println(patches.get(0).length);
        //System.out.println(App.splits.getTrainingDataset().numInstances());

        /*
        Taking training dataset split, sampling images from these and getting all the patches
         */
        ArrayList<float[]> patches = new ArrayList<>();
        for (FImage image: GroupedUniformRandomisedSampler.sample(App.splits.getTrainingDataset(),30)){
            patches.addAll(getPatches(image, 2, 4));
        }

        //making assigner, extractor and liblinearannotator with these patches and the training dataset
        HardAssigner<float[], float[], IntFloatPair> assigner = KMeansClustering(patches.toArray(new float[patches.size()][]));
        BOVWExtractor extractor = new BOVWExtractor(assigner);
        LiblinearAnnotator<FImage, String> annotator = new LiblinearAnnotator<FImage, String>(
               extractor , LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001 );
        annotator.train(App.splits.getTrainingDataset() );

        //writing to run2.txt with the classification result and the name of the file.
        try {
            PrintWriter printWriter = new PrintWriter(new File("run2.txt"));
            for (int i = 0; i < App.testingData.size(); i++) {
                printWriter.println(App.testingData.getID(i).substring(8) + " " + annotator.classify(App.testingData.get(i)).getPredictedClasses().toArray()[0].toString());
            }
            printWriter.flush();
            printWriter.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        //code used for testing purposes
        /*ClassificationEvaluator<CMResult<String>, String, FImage> eval =
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

    /**
     * This function creates rectangles and then extracts the region of interest using these rectangles.
     * @param inputImage : FImage that you wish to extract the patches from
     * @param interval : Int how often stepwise x and y do you wish to sample these patches from
     *                 (smaller intervals result in longer training times typically)
     * @param patchsize : Int how large are the patches done via a X by X size resulting a patch that is X squared area
     * @return List of float arrays, each float array represents a patch. Contains all the patches in the image.
     */
    private static List<float[]> getPatches(FImage inputImage, int interval, int patchsize){

        RectangleSampler rectSampler = new RectangleSampler(inputImage, interval, interval, patchsize, patchsize);
        List<Rectangle> rectangles = rectSampler.allRectangles();

        ArrayList<float[]> vectors = new ArrayList<>();

        for (Rectangle rectangle: rectangles){
            FImage imagePatch = inputImage.extractROI(rectangle);
            //mean centering and normalizing the patch using the method from run 1
            imagePatch = imagePatch.subtract(Run1.averageFloat(imagePatch));
            imagePatch = imagePatch.normalise();
            vectors.add(imagePatch.getFloatPixelVector());
        }
        //float[][] arr = new float[vectors.size()][];
        return vectors;
    }

    /**
     * This function will cluster the patches and then return the hard assigner from this result
     * @param floatPatches the float 2d array of the patches you wish to cluster
     * @return HardAssigner of used later for the extractor.
     */
    private static HardAssigner<float[], float[], IntFloatPair> KMeansClustering(float[][] floatPatches){
        System.out.println(floatPatches.length);
        final FloatKMeans kMeans =  FloatKMeans.createExact(1000);
        FloatCentroidsResult result = kMeans.cluster(floatPatches);
        HardAssigner<float[], float[], IntFloatPair> hardAssigner =  result.defaultHardAssigner();
        return hardAssigner;
    }

    /**
     * Class made to be our Bag of visual words extractor, with a way of extracting features (inspired by CH12)
     */
    static class BOVWExtractor implements FeatureExtractor<SparseIntFV, FImage> {
        HardAssigner<float[], float[], IntFloatPair> assigner;

        BOVWExtractor(HardAssigner<float[], float[], IntFloatPair> assigner){
            this.assigner = assigner;
        }

        /**
         * Overrides the extraFeature method in the interface FeatureExtractor
         * @param object this is the object we want to extract the features from (in our case this would be FImages.)
         * @return returns a SparseIntFV from BagOfVisualWords' method aggregrateVectorsRaw from our patchlist.
         */
        @Override
        public SparseIntFV extractFeature(FImage object) {
            List<float[]> patchList = getPatches(object, 2,4);
            BagOfVisualWords<float[]> bagOfVisualWords = new BagOfVisualWords<float[]>( assigner );
            return bagOfVisualWords.aggregateVectorsRaw( patchList );
        }

    }
}
