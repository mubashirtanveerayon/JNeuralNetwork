public class Main {
    public static void main(String[] args) {
//         NeuralNetwork nn = new NeuralNetwork(2,1);
//         nn.addLayer(30);
//         nn.addLayer(20);
//         //nn.addLayer(1);
//         nn.initialize();
//         //System.out.println(args.length);
//         Matrix input = Matrix.fromArray(new float[]{1,0});
//         //nn.feedForward(input).print();
//          for(int i=0;i<10000;i++){
//              nn.train(Matrix.fromArray(new float[]{1,0}),Matrix.fromArray(new float[]{1}));
//              nn.train(Matrix.fromArray(new float[]{0,1}),Matrix.fromArray(new float[]{1}));
//              nn.train(Matrix.fromArray(new float[]{1,1}),Matrix.fromArray(new float[]{0}));
//              nn.train(Matrix.fromArray(new float[]{0,0}),Matrix.fromArray(new float[]{0}));
//          }
//         nn.feedForward(input).print();
//         nn.feedForward(Matrix.fromArray(new float[]{1,1})).print();
//         System.out.println(nn.export());

        NeuralNetwork nn =  NeuralNetwork.createModel("src/model");
        System.out.println(nn);


    }
}
