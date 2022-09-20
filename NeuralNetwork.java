import java.util.ArrayList;
import java.nio.file.*;

public class NeuralNetwork{

    ArrayList<Matrix> weights,biases;
    boolean initialized;
    int numInputs,numOutputs,numLayers=0;
    float learningRate = 0.1f;

    public NeuralNetwork(ArrayList<Matrix>w,ArrayList<Matrix>b){
        if(w.size() != b.size()){
            System.out.println("invalid model");
            return;
        }
        weights = new ArrayList<>(w);
        biases = new ArrayList<>(b);
        numInputs = w.get(0).cols;
        numOutputs = w.get(w.size()-1).rows;
        numLayers = w.size()-1;
    }

    public NeuralNetwork(int numI,int numO){
        numInputs = numI;
        numOutputs = numO;

        weights = new ArrayList<>();
        biases = new ArrayList<>();
    }

    public void initialize(){
        initialized = numLayers>0;
        if(initialized){
          System.out.println("Initialized neural network with "+numLayers+" hidden layers, "+numInputs+" inputs and "+numOutputs+" outputs!");  
        }
    }

    public boolean export() {
        if(!initialized)return false;
        try{
        Path directory = Paths.get("model");
        if(!Files.exists(directory)){
            Files.createDirectory(directory);
        }

        String wFileExt = "weight";
        String bFileExt = "bias";

        for(int i=0;i<=numLayers;i++){
            String wStrFormat = weights.get(i).toString();
            Path wFile = Paths.get("model/layer"+String.valueOf(i)+"."+wFileExt);
            Files.write(wFile,wStrFormat.getBytes());
            String bStrFormat = biases.get(i).toString();
            Path bFile = Paths.get("model/layer"+String.valueOf(i)+"."+bFileExt);
            Files.write(bFile,bStrFormat.getBytes());
        }
        }catch(Exception e){
            e.printStackTrace();
            return false;
        }
        return true;

    }

    public void addLayer(int numNodes){
        if(initialized)return;
        if(numLayers==0){
            weights.add(new Matrix(numNodes,numInputs).randomize());
            weights.add(new Matrix(numOutputs,numNodes).randomize());
            biases.add(new Matrix(numNodes,1).randomize());
            biases.add(new Matrix(numOutputs,1).randomize());
        }else{
            weights.remove(weights.size()-1);
            biases.remove(biases.size()-1);
            weights.add(new Matrix(numNodes,weights.get(weights.size()-1).rows).randomize());
            biases.add(new Matrix(numNodes,1).randomize());
            biases.add(new Matrix(numOutputs,1).randomize());
            weights.add(new Matrix(numOutputs,numNodes).randomize());
        }
        numLayers++;
    }

    public Matrix feedForward(Matrix input){
        if(!initialized)return null;
        Matrix result = input;
        for(int i=0;i<weights.size();i++){
            result = Matrix.mult(weights.get(i),result).addSelf(biases.get(i)).sigmoid();
        }
        return result;
    }

    public void train(final Matrix input,final Matrix target){
        //feedforward
        Matrix result = input.copy();
        ArrayList<Matrix> layerOutputs = new ArrayList<>();
        layerOutputs.add(input.copy());
        for(int i=0;i<weights.size();i++){
            result = Matrix.mult(weights.get(i),result).addSelf(biases.get(i)).sigmoid();
            layerOutputs.add(result.copy());
        }

        //backpropagation
        Matrix outputError = Matrix.add(target,Matrix.mult(result,-1));
        Matrix layerError = outputError;

        for(int i=layerOutputs.size()-1;i>=1;i--){
            Matrix gradient = layerError.copy().multSelf(learningRate).multElementWiseSelf(layerOutputs.get(i).dsigmoidFromPreviousSigmoid());;

            Matrix wDT = Matrix.mult(gradient,Matrix.transpose(layerOutputs.get(i-1)));
            Matrix w = weights.get(i-1);

            layerError = Matrix.mult(Matrix.transpose(w),layerError);
            w.addSelf(wDT);
            Matrix b = biases.get(i-1);
            b.addSelf(gradient);
        }

    }


    public static NeuralNetwork createModel(String modelPath) {
        ArrayList<Matrix> weights = new ArrayList<>();
        ArrayList<Matrix> biases = new ArrayList<>();
        try{
            for(Object o:Files.list(Paths.get(modelPath)).toArray()){
                if(((Path)o).getFileName().toString().endsWith("weight")){
                    weights.add(new Matrix(Files.readAllLines((Path)o)));
                }else if(((Path)o).getFileName().toString().endsWith("bias")){
                    biases.add(new Matrix(Files.readAllLines((Path)o)));
                }
            }
        }catch(Exception e){
            e.printStackTrace();
            return null;
        }
        return new NeuralNetwork(weights,biases);
    }


}
