public class Main {
    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(2,1);
        nn.addLayer(3);
        //nn.addLayer(5);
        //nn.addLayer(1);
        nn.initialize();
        //System.out.println(args.length);
        Matrix input = Matrix.fromArray(new float[]{Integer.parseInt(args[0]),Integer.parseInt(args[1])});
        //nn.feedForward(input).print();
         for(int i=0;i<50000;i++){
             nn.train(Matrix.fromArray(new float[]{1,0}),Matrix.fromArray(new float[]{1}));
             nn.train(Matrix.fromArray(new float[]{0,1}),Matrix.fromArray(new float[]{1}));
             nn.train(Matrix.fromArray(new float[]{1,1}),Matrix.fromArray(new float[]{1}));
             nn.train(Matrix.fromArray(new float[]{0,0}),Matrix.fromArray(new float[]{0}));
         }
         nn.feedForward(input).print();
    }
}
