@echo off
IF "%1%" == "all" (
    echo running all stages 
    
    word2vec 
    echo running word2vec on each trait ...
    python src\word2vec\run.py
    echo ----------------------
    echo running word2vec on dataset ...
    python src\word2vec\all_run.py

    @REM tokenization
    echo tokenization process ...
    python src\tokenization\trainer.py
    echo analyzing process ...
    python src\tokenization\analyzer.py

    @REM feature enginnering
    echo feature enginnering has been started ...
    python src\feature_engineering\main.py
    echo analyzing process ...

    @REM model architecture
    echo running just NN word2vec combined with ParsBERT model ...
    python src\model_architecture\combination_parsbert_word2vec.py

) ELSE IF "%1%" == "word2vec" (
    echo running word2vec on each trait ...
    python src\word2vec\run.py
    echo ----------------------
    echo running word2vec on dataset ...
    python src\word2vec\all_run.py
) ELSE IF "%1%" == "tokenization" (
    echo tokenization process ...
    python src\tokenization\trainer.py
    echo analyzing process ...
    python src\tokenization\analyzer.py
) ELSE IF "%1%" == "feat_engineering" (
    echo feature enginnering has been started ...
    python src\feature_engineering\main.py
    echo analyzing process ...
) ELSE IF "%1%" == "model_architecture" (
    @REM model architecture
    echo running just NN word2vec combined with ParsBERT model ...
    python src\model_architecture\combination_parsbert_word2vec.py
) ELSE IF "%1%" == "report" (
    echo making latex
    cd report\
    xelatex -output-directory=..\ReportOutput main.tex > nul
    xelatex -output-directory=..\ReportOutput main.tex
    cd ..
) ELSE (
    echo you should pick between "all", "report", "tokenization", "model_architecture", "word2vec"
)
