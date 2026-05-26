# MARL_TTA_HSI
A Generalizable Human-on-the-Loop Threat-Aware MARL Framework for Multi-agent Confrontation

## Project Overview
A human-on-the-loop, threat-aware MARL framework with TTA and HSI module for efficient and generalizable multi-UAV confrontation in high-fidelity Harfang3D simulations.

## Architecture

### Framework
We develop the human-on-the-loop threat-aware MARL framework that comprises three core modules: (1) the Multi-Agent Reinforcement Learning (MARL) module, (2) the Target Threat Assessment (TTA) module, and (3) the Human-Swarm Interaction (HSI) module. These modules operate synergistically to achieve improved decision-making in MACS.
<img width="1671" height="953" alt="image" src="https://github.com/user-attachments/assets/1e57a014-dd0a-4aef-9f68-24ba15cc237e" />


### Human-Swarm Interaction Interface
The human-swarm interaction interface establishes a bidirectional TCP/IP communication channel with both the MARL algorithm and the MACS, enabling real-time data exchange. This interface comprises two primary components: (1) an interactive control panel and (2) a 3D situational display. The 3D visualization module provides an intuitive representation of the confrontation scenario.
<img width="1754" height="993" alt="image" src="https://github.com/user-attachments/assets/2a006f7b-7ada-443b-8223-9d3485ffd711" />

### Repository Structure


### Modules


### Quick Start


### Results


### Acknowledgements / References
Simulation backend is based on the Harfang3D Dog-Fight Sandbox. Please cite:
```bibtex
@misc{2210.07282,
  Author = {Muhammed Murat Özbek,  Süleyman Yıldırım,  Muhammet Aksoy, Eric Kernin and Emre Koyuncu},
  Title = {Harfang3D Dog-Fight Sandbox: A Reinforcement Learning Research Platform for the Customized Control Tasks of Fighter Aircrafts},
  publisher = {arXiv},
  doi = {10.48550/ARXIV.2210.07282},
  Year = {2022},
  Eprint = {arXiv:2210.07282},
}
```

