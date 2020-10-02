//	A deep neural networks class in Java WITH VARIABLE WIDTH AND DEPTH.
//
//	Uses arrays to save partial derivatives for each perceptron. The time
//	complexity of the back propagation algorithm is linear in terms of depth
//	and polynomial in terms of width.
//
//	The width of each layer is variable, and no file IO methods were added
//	since this version was designed to be extended by a separate IO class,
//	cf. github.com/MattMattL/OpenSteve
//

import java.util.Random;

public class DeepNNet
{
	public final int NET_DEPTH;
	public final int NET_MAX_WIDTH;
	public final int NET_IN;
	public final int NET_OUT;
	private final int NET_WIDTH[];

	public double vectorIn[];
	protected double vectorOut[];
	public double vectorDesired[];

	protected double weights[][][];
	private double products[][];
	private double sigmoid[][];
	private double delta[][];

	private final Random random;

	public DeepNNet(int netIn, int netDepth, int netOut)
	{
		this.random = new Random(System.currentTimeMillis());

		this.NET_IN = netIn;
		this.NET_OUT = netOut;
		this.NET_DEPTH = netDepth;
		this.NET_MAX_WIDTH = Math.max(this.NET_IN, this.NET_OUT);
		this.NET_WIDTH = new int[NET_DEPTH + 1];

		memoryInit();
		vectorInit();
	}

	private void memoryInit()
	{
		vectorIn = new double[NET_IN];
		vectorOut = new double[NET_OUT];
		vectorDesired = new double[NET_OUT];

		weights = new double[NET_DEPTH][NET_MAX_WIDTH][NET_MAX_WIDTH];
		products = new double[NET_DEPTH][NET_MAX_WIDTH];
		sigmoid = new double[NET_DEPTH + 1][NET_MAX_WIDTH];

		delta = new double[NET_DEPTH][NET_MAX_WIDTH];

		NET_WIDTH[0] = NET_IN;
		NET_WIDTH[NET_DEPTH] = NET_OUT;

		for(int l=1; l<NET_DEPTH; l++)
			NET_WIDTH[l] = (this.NET_IN + this.NET_OUT) / 2; // temporary initialisation method
	}

	private void vectorInit()
	{
		for(int i=0; i<NET_IN; i++)
			vectorIn[i] = random.nextDouble();

		for(int i=0; i<NET_OUT; i++)
			 vectorDesired[i] = random.nextDouble();

		for(int l=0; l<NET_DEPTH; l++)
		{
			for(int i=0; i<NET_WIDTH[l]; i++)
				for(int j=0; j<NET_WIDTH[l+1]; j++)
					weights[l][i][j] = random.nextDouble() * 2 - 1;
		}
	}

	public void nnRunFeedForward()
	{
		int in, out, l, i, j;

		// copy input vector
		for(in=0; in<NET_IN; in++)
			sigmoid[0][in] = vectorIn[in];

		// run neural networks
		for(l=0; l<NET_DEPTH; l++)
		{
			for(j=0; j<NET_WIDTH[l+1]; j++)
			{
				products[l][j] = 0;

				for(i=0; i<NET_WIDTH[l]; i++)
					products[l][j] += sigmoid[l][i] * weights[l][i][j];
			}

			for(j=0; j<NET_WIDTH[l+1]; j++)
				sigmoid[l+1][j] = 1.0/(1 + Math.exp(-products[l][j]));
		}

		// save output vector
		for(out=0; out<NET_OUT; out++)
			vectorOut[out] = sigmoid[NET_DEPTH][out];
	}

	public void nnRunBackprop()
	{
		int layer, out, i, j, k;

		// calculate delta
		for(out=0; out<NET_OUT; out++)
			delta[NET_DEPTH-1][out] = 0;

		for(out=0; out<NET_OUT; out++)
			delta[NET_DEPTH-1][out] += (vectorDesired[out] - vectorOut[out]) * vectorOut[out] * (1 - vectorOut[out]);

		for(layer=NET_DEPTH - 2; layer>=0 ; layer--)
		{
			for(j=0; j<NET_WIDTH[layer+1]; j++)
			{
				delta[layer][j] = 0;

				for(k=0; k<NET_WIDTH[layer+2]; k++)
					delta[layer][j] += delta[layer+1][k] * weights[layer+1][j][k] * sigmoid[layer+1][j] * (1 - sigmoid[layer+1][j]);
			}
		}
		// adjust the weights
		for(layer=0; layer<NET_DEPTH; layer++)
		{
			for(j=0; j<NET_WIDTH[layer+1]; j++)
				for(i=0; i<NET_WIDTH[layer]; i++)
					weights[layer][i][j] += delta[layer][j] * sigmoid[layer][i];
		}
	}
}
