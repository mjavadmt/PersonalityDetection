open System
open System.IO
open FSharp.Data
open FSharp.Json
open Data.DatasetGeneration.SemiAutoLabel.Common.Cleaners
open Data.DatasetGeneration.SemiAutoLabel.Common.Data
open Data.Common.DataDirectoryPath

let tweetFiles =
    DirectoryInfo($"{INTERIM}/{DATA_NAME}/tweets")
        .GetFiles()
    |> Seq.map (fun f -> (f.Name.Replace(".json", ""), f))
    |> Map.ofSeq


let bioFiles =
    DirectoryInfo($"{INTERIM}/{DATA_NAME}/bios")
        .GetFiles()
    |> Seq.map (fun f -> (f.Name.Replace(".txt", ""), f))
    |> Map.ofSeq


let mbtiLabelsHandChecked =
    JsonValue
        .Load($"{INTERIM}/{DATA_NAME}/mbti-labels-handchecked.json")
        .Properties()
    |> Seq.map (fun (k, v) -> (k, v.AsString()))


let readBio handle =
    File.ReadAllText(bioFiles.[handle].FullName)
    |> commonClean
    |> aggressivelyClean


let readTweets handle =
    JsonValue
        .Load(tweetFiles.[handle].FullName)
        .AsArray()
    |> Array.Parallel.map (
        (fun row -> row.GetProperty("tweet").AsString())
        >> commonClean
        >> aggressivelyClean
    )
    |> Array.filter (String.IsNullOrWhiteSpace >> not)


let dataset() =
    mbtiLabelsHandChecked
    |> Array.ofSeq
    |> Array.filter
        (fun (handle, _) ->
            tweetFiles.ContainsKey(handle)
            && bioFiles.ContainsKey(handle))
    |> Array.Parallel.map
        (fun (handle, label) ->
            {| Handle = handle
               Bio = readBio handle
               Tweets = readTweets handle
               MbtiLabel = label |})


[<EntryPoint>]
let main _ =
    dataset()
    |> Json.serialize
    |> fun json -> File.WriteAllText($"{PROCESSED}/{DATA_NAME}/dataset-hand-labeled.json", json)

    0
