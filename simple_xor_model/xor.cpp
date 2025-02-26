#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

// Activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the sigmoid function
double sigmoid_derivative(double output) {
    return output * (1.0 - output);
}

// Training data for XOR
vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
vector<double> targets = {0, 1, 1, 0};

// Network parameters
double hidden_weights[2][2];
double hidden_bias[2];
double output_weights[2];
double output_bias;

// Initialize weights and biases with random values between -1 and 1
void initialize_network() {
    srand(time(0));
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            hidden_weights[i][j] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        }
        hidden_bias[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }
    for (int i = 0; i < 2; ++i) {
        output_weights[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }
    output_bias = (double)rand() / RAND_MAX * 2.0 - 1.0;
}

int main() {
    initialize_network();

    double learning_rate = 0.1;
    int max_epochs = 10000;
    double error_threshold = 0.001;

    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        double total_error = 0.0;

        for (int i = 0; i < 4; ++i) {
            // Input and target
            double x0 = inputs[i][0];
            double x1 = inputs[i][1];
            double target = targets[i];

            // Forward pass
            double h0 = x0 * hidden_weights[0][0] + x1 * hidden_weights[0][1] + hidden_bias[0];
            double h0_act = sigmoid(h0);
            double h1 = x0 * hidden_weights[1][0] + x1 * hidden_weights[1][1] + hidden_bias[1];
            double h1_act = sigmoid(h1);

            double output = h0_act * output_weights[0] + h1_act * output_weights[1] + output_bias;
            output = sigmoid(output);

            // Error calculation
            double error = target - output;
            total_error += error * error;

            // Backpropagation
            // Output layer delta
            double delta_output = error * sigmoid_derivative(output);

            // Hidden layer deltas
            double delta_hidden0 = delta_output * output_weights[0] * sigmoid_derivative(h0_act);
            double delta_hidden1 = delta_output * output_weights[1] * sigmoid_derivative(h1_act);

            // Update output weights and bias
            output_weights[0] += learning_rate * delta_output * h0_act;
            output_weights[1] += learning_rate * delta_output * h1_act;
            output_bias += learning_rate * delta_output;

            // Update hidden layer weights and biases
            hidden_weights[0][0] += learning_rate * delta_hidden0 * x0;
            hidden_weights[0][1] += learning_rate * delta_hidden0 * x1;
            hidden_bias[0] += learning_rate * delta_hidden0;

            hidden_weights[1][0] += learning_rate * delta_hidden1 * x0;
            hidden_weights[1][1] += learning_rate * delta_hidden1 * x1;
            hidden_bias[1] += learning_rate * delta_hidden1;
        }

        total_error /= 4.0; // Average error over all samples

        if (epoch % 1000 == 0) {
            cout << "Epoch " << epoch << " - Error: " << total_error << endl;
        }

        if (total_error < error_threshold) {
            cout << "Training stopped at epoch " << epoch << " with error " << total_error << endl;
            break;
        }
    }

    // Test the trained network
    cout << "\nTesting the trained network:" << endl;
    for (int i = 0; i < 4; ++i) {
        double x0 = inputs[i][0];
        double x1 = inputs[i][1];

        double h0 = x0 * hidden_weights[0][0] + x1 * hidden_weights[0][1] + hidden_bias[0];
        double h0_act = sigmoid(h0);
        double h1 = x0 * hidden_weights[1][0] + x1 * hidden_weights[1][1] + hidden_bias[1];
        double h1_act = sigmoid(h1);

        double output = h0_act * output_weights[0] + h1_act * output_weights[1] + output_bias;
        output = sigmoid(output);

        cout << x0 << " XOR " << x1 << " = " << round(output) << " (" << output << ")" << endl;
    }

    return 0;
}
