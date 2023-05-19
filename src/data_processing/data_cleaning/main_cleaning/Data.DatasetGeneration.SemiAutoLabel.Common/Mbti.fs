module Data.DatasetGeneration.SemiAutoLabel.Common.Mbti

let mbtiLabels =
    query {
        for ie in [ "I"; "E" ] do
            for ns in [ "N"; "S" ] do
                for tf in [ "T"; "F" ] do
                    for pj in [ "P"; "J" ] do
                        select $"{ie}{ns}{tf}{pj}"
    }
