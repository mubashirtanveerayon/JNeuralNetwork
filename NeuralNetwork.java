import java.util.ArrayList;

public class NeuralNetwork{

    public ArrayList<Matrix> weights,biases;
    boolean initialized;
    int numInputs,numOutputs,numLayers=0;
    float learningRate = 0.1f;


    public NeuralNetwork(int numI,int numO){
        numInputs = numI;
        numOutputs = numO;

        weights = new ArrayList<>();
        biases = new ArrayList<>();
    }

    public void initialize(){
        initialized = true;
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
            weights.add(new Matrix(numNodes,weights.get(weights.size()-1).rows));
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
            /*try{
            result = Matrix.mult(weights.get(i),result).addSelf(biases.get(i)).sigmoid();
            }catch(Exception e){
                System.out.println(i);
                weights.get(i).print();
                result.print();
                System.exit(0);
            }*/
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

}
