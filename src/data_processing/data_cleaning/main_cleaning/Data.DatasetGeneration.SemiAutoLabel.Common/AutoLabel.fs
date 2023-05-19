module Data.DatasetGeneration.SemiAutoLabel.Common.AutoLabel

open System
open Data.DatasetGeneration.SemiAutoLabel.Common.Mbti


let extractLabelsFromText (text: string) =
    mbtiLabels
    |> Seq.filter (fun it -> text.Contains(it, StringComparison.InvariantCultureIgnoreCase))


let extractLabelsFromBio (cleanBio: string) =
    cleanBio |> extractLabelsFromText |> List.ofSeq


let extractIndicatorTweets (cleanTweets: list<string>) =
    [ 0 .. cleanTweets.Length - 1 ]
    |> List.filter
        (fun i ->
            (mbtiLabels
             |> Seq.exists
                 (fun l ->
                     cleanTweets.[i]
                         .Contains(l, StringComparison.InvariantCultureIgnoreCase))))
    |> List.map (fun i -> cleanTweets.[i])


let extractLabelsFromIndicatorTweets (cleanIndicatorTweets: list<string>) =
    cleanIndicatorTweets
    |> Seq.collect extractLabelsFromText
    |> Seq.distinct
    |> List.ofSeq


let extractLabel (cleanBio: string, cleanIndicatorTweets: list<string>) =
    let labelsFromTweets =
        cleanIndicatorTweets
        |> extractLabelsFromIndicatorTweets

    let labelsFromBio = cleanBio |> extractLabelsFromBio

    match labelsFromBio.Length with
    | 0 ->
        match labelsFromTweets.Length with
        | 0 -> None
        | 1 -> Some labelsFromTweets.[0]
        | _ -> None
    | 1 ->
        match labelsFromTweets.Length with
        | 0 -> Some labelsFromBio.[0]
        | 1 -> if labelsFromBio.[0] = labelsFromTweets.[0] then Some labelsFromBio.[0] else None
        | _ -> None
    | _ -> None
