open System
open System.IO
open FSharp.Data
open FSharp.Json
open Data.DatasetGeneration.SemiAutoLabel.Common.Cleaners
open Data.DatasetGeneration.SemiAutoLabel.Common.Data
open Data.DatasetGeneration.SemiAutoLabel.Common.AutoLabel
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


let handles =
    Set.intersect
        (bioFiles |> Map.toSeq |> Seq.map fst |> Set.ofSeq)
        (tweetFiles
         |> Map.toSeq
         |> Seq.map fst
         |> Set.ofSeq)


let readBio handle =
    File.ReadAllText(bioFiles.[handle].FullName)


let readTweets handle =
    JsonValue
        .Load(tweetFiles.[handle].FullName)
        .AsArray()
    |> Array.map (fun row -> row.GetProperty("tweet").AsString())


let cleanBio handle = handle |> readBio |> commonClean


let cleanTweets handle =
    handle
    |> readTweets
    |> Array.map commonClean
    |> Array.filter (String.IsNullOrWhiteSpace >> not)
    |> Array.toList


let cleanDataset () =
    handles
    |> Array.ofSeq
    |> (fun it ->
        printfn $"clean:\n\t initial count: {it.Length} users"
        it)
    |> Array.Parallel.map
        (fun h ->
            let b = cleanBio h
            let t = cleanTweets h
            let it = t |> extractIndicatorTweets
            let l = extractLabel (b, it)

            match l with
            | None -> Option.None
            | Some l -> Some(h, b, t, it, l))
    |> Array.Parallel.choose id
    |> (fun it ->
        printfn $"clean:\n\t final count: {it.Length} users"
        it)
    |> Array.Parallel.map
        (fun (h, b, t, it, l) ->
            {| Handle = h
               Bio = b
               Tweets = t
               IndicatorTweets = it
               MbtiLabel = l |})


let aggressivelyCleanDataset () =
    handles
    |> Array.ofSeq
    |> (fun it ->
        printfn $"aggressive clean:\n\t initial count: {it.Length} users"
        it)
    |> Array.Parallel.map
        (fun h ->
            let cb = cleanBio h
            let ct = cleanTweets h
            let it = ct |> extractIndicatorTweets
            let l = extractLabel (cb, it)

            let b = cb |> aggressivelyClean

            let t =
                ct
                |> List.map aggressivelyClean
                |> List.filter (String.IsNullOrWhiteSpace >> not)

            match l with
            | None -> Option.None
            | Some l -> Some(h, b, t, it, l))
    |> Array.Parallel.choose id
    |> (fun it ->
        printfn $"aggressive clean:\n\t final count: {it.Length} users"
        it)
    |> Array.Parallel.map
        (fun (h, b, t, it, l) ->
            {| Handle = h
               Bio = b
               Tweets = t
               IndicatorTweets = it
               MbtiLabel = l |})

[<EntryPoint>]
let main _ =
    let stopWatch = System.Diagnostics.Stopwatch()
    stopWatch.Start()
    printfn $"{stopWatch.Elapsed.TotalSeconds}"

    cleanDataset ()
    |> Json.serialize
    |> fun json -> File.WriteAllText($"{PROCESSED}/{DATA_NAME}/clean-dataset.json", json)

    printfn $"{stopWatch.Elapsed.TotalSeconds}"


    aggressivelyCleanDataset ()
    |> Json.serialize
    |> fun json -> File.WriteAllText($"{PROCESSED}/{DATA_NAME}/aggressively-clean-dataset.json", json)

    stopWatch.Stop()
    printfn $"{stopWatch.Elapsed.TotalSeconds}"
    0
