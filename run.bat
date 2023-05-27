@echo off
IF "%1%" == "all" (
    python src\data_processing\main.py %1
    IF "%2%" == "analyze" (
        python src\data_analyzing\main.py
    )
    cd report\
    xelatex -output-directory=..\ReportOutput main.tex
) ELSE IF "%1%" == "clean" (
    python src\data_processing\main.py %1
) ELSE IF "%1%" == "collect" (
    python src\data_processing\main.py %1
) ELSE IF "%1%" == "tokenize" (
    python src\data_processing\main.py %1
) ELSE IF "%1%" == "analyze" (
    python src\data_analyzing\main.py
    cd report\
    xelatex -output-directory=..\ReportOutput main.tex
) ELSE (
    echo you are not entering correct argument choose between "all", "clean", "collect", "tokenize", "analyze"
    echo if you use analyze argument as the first argument it will analyze dataset and create report file
    echo also you can use analyze as the second argument besides all argument that will do both data_processing and data_analyzing
)
