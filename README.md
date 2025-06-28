# sos_app

A new Flutter project.

## Getting Started

This project is a starting point for a Flutter application.

A few resources to get you started if this is your first Flutter project:

- [Lab: Write your first Flutter app](https://docs.flutter.dev/get-started/codelab)
- [Cookbook: Useful Flutter samples](https://docs.flutter.dev/cookbook)

For help getting started with Flutter development, view the
[online documentation](https://docs.flutter.dev/), which offers tutorials,
samples, guidance on mobile development, and a full API reference.


The application’s architecture consists of four main layers: the Presentation Layer (mobile interface), the Service, the Processing Layer (AI-driven accident classification engine), and the Data Layer (SQLite for local storage with cloud redundancy). The core of the system is an AI classification engine that processes continuous audio streams from the device’s microphone. It extracts Mel-Frequency Cepstral Coefficients (MFCCs) from the audio, which are then analyzed by a Random Forest machine learning model trained to identify accident-related sounds. 
