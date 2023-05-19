open System
open FSharp.Data
open Data.Common.DataDirectoryPath


[<Literal>]
let dataName = 
    "semi-auto-label"

[<Literal>]
let aggressivelyCleanDatasetPath = 
    PROCESSED + "/" + dataName + "/aggressively-clean-dataset.json"

type AggressivelyCleanDataset =
    JsonProvider<aggressivelyCleanDatasetPath>

let aggressivelyCleanDataset = 
    AggressivelyCleanDataset.Load(aggressivelyCleanDatasetPath)

let aggressivelyCleanDatasetByHandle =
    aggressivelyCleanDataset
    |> Seq.map (fun it -> (it.Handle, it))
    |> Map.ofSeq

[<Literal>]
let datasetWithoutFilterPath = 
    PROCESSED + "/" + dataName + "/dataset-without-filter.json"

type DatasetWithoutFilter =
    JsonProvider<datasetWithoutFilterPath>

let datasetWithoutFilter = 
    DatasetWithoutFilter.Load(datasetWithoutFilterPath)

let datasetWithoutFilterByHandle =
    datasetWithoutFilter
    |> Seq.map (fun it -> (it.Handle, it))
    |> Map.ofSeq

let commonHandles =
    let acd =  
        aggressivelyCleanDataset
        |> Seq.map (fun it -> it.Handle)
        |> Set.ofSeq

    let dwf =
        datasetWithoutFilter
        |> Seq.map (fun it -> it.Handle)
        |> Set.ofSeq

    Set.intersect acd dwf

let handlesWithUnmatchedLabels =
    commonHandles
    |> Seq.filter (fun handle -> aggressivelyCleanDatasetByHandle.[handle].MbtiLabel <> datasetWithoutFilterByHandle.[handle].MbtiLabel)
    |> Seq.map (fun handle -> $"{handle}: {aggressivelyCleanDatasetByHandle.[handle].MbtiLabel} <> {datasetWithoutFilterByHandle.[handle].MbtiLabel}")
    |> String.concat "\n"

[<EntryPoint>]
let main _ =
    printfn $"{handlesWithUnmatchedLabels}"
    0
