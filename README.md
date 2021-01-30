# Solução de Nicolas Melo

Computer Vision Engineer at Dod Vision & Pega Placa
IN: https://www.linkedin.com/in/n%C3%ADcolas-melo-bsb/
Telegram: https://t.me/Nicolas_melo_CV

# Resultados alcançados
34% de similaridade com as imagens originais (podem ser achadas aqui https://www.kaggle.com/c/denoising-dirty-documents/data) no teste de cleanup
99.4% de precisão nos Centroid do onde está wally (existem formas mais simples de resolver isso em comparação ao que utilizei)

# Cleanup
Utilze um detector de texto (pode ser o EAST ou CTPN) para o detectar o eixo Y de cada palavra e fazer um deskew da formatação de fonte itálica para normal.
Para métricas utilize o similarity image do scikit-image

# wally
Model utilizado Faster-Rcnn 640x640 tensorflow object detection API
Acredito que um canny edge e minimum rec area consiga resolver o problema do wally
Se optar pela detecção de objetos recomendo Tiny Yolo v3 ou v4
ps: Não possuo o Ground Truth do CentroID do wally

# Do Processo Seletivo: 
Os resultados são parcialmente ignorados o que mais vale é a sua explicação sobre a solução e algumas perguntas adicionais de experiência e conhecimentos das ferramentas em geral de CV.
(para perguntas adicionais envie um email: nicolasfmelo@gmail.com)

# Descrições adicionais:
A prova foi feita as pressas então o código está bem desorganizado.
As instruções do teste estão logo abaixo, foi editada da original pois só respondi as questões de Visão Computacional. 
Desafio completo em: (https://github.com/nuveo/cv-test)

# Artificial Intelligence test

This repository has several tests that comply with the Computer Vision, Natural Language Processing and Machine Learning expertise. Each test has its own set of files, parameters, instructions and strategies to be solved. Therefore, choose them wisely.

# Tests

The following tests are given:

1. (:muscle:) **Document Cleanup** (Computer Vision) 
3. (:muscle:) **Where's Wally?** (Computer Vision) 

Where the level of difficulty can be (roughly) defined such as:

- :ok_hand: : It is regular challenge that should be fine for most of AI enthusiasts.
- :muscle: : Increase the level of complexity and requires more experience on the AI field
- :punch: : It is a good challenge for AI specialists that are both curious and have great familiriaty in the field

The instructions to each problem are described in separated README files in each folder.

# Instructions

Please, develop a script or computer program using the programming language of your choice to solve at least **two** of these tests, where the candidate is free to choose any of them. We are aware of the difficulty associated with each problem, but all creativeness, reasoning strategy, details on code documentation, code structure and accuracy will be used to evaluate the candidate performance. So make sure the presented code reflects your knowledge as much as possible!

We expect that a solution could be achieved within a reasonable time-period, considering a few days, so be free to use the time as best as possible. We understand that you may have a tight schedule, therefore do not hesitate to contact us for any further request :+1:.

## Datasets

All the datasets are located into a single compressed file in [this link](https://drive.google.com/file/d/1hAyIlI8NjvG8dkm8hmAe3URf2v9Tlfz-/view?usp=sharing). 

    Note that the file is quite big (~1 Gb), but we believe that a few minutes could deal with it

