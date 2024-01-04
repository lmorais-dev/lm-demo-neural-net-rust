use std::u16::MAX;

use activations::SIGNOID;
use network::Network;

mod matrix;
mod network;
mod activations;

fn main() {
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0]
    ];

	let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    let mut network = Network::new(vec![2, 3, 3, 1], 0.1, SIGNOID);
    network.train(inputs.clone(), targets.clone(), MAX);

    println!("{:?}", network.feed_forward(vec![0.0, 0.0]).unwrap());
	println!("{:?}", network.feed_forward(vec![0.0, 1.0]).unwrap());
	println!("{:?}", network.feed_forward(vec![1.0, 0.0]).unwrap());
	println!("{:?}", network.feed_forward(vec![1.0, 1.0]).unwrap());

    network.train(inputs, targets, MAX);

    println!("{:?}", network.feed_forward(vec![0.0, 0.0]).unwrap());
	println!("{:?}", network.feed_forward(vec![0.0, 1.0]).unwrap());
	println!("{:?}", network.feed_forward(vec![1.0, 0.0]).unwrap());
	println!("{:?}", network.feed_forward(vec![1.0, 1.0]).unwrap());
}
