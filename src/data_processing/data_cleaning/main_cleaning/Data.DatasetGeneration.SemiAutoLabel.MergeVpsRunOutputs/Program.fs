open System
open System.IO
open Data.Common.DataDirectoryPath
open Data.DatasetGeneration.SemiAutoLabel.Common.Data


let vpsRunOutputDirs =
    [1; 2]
    |> Seq.map (fun it -> $"{INTERIM}/{DATA_NAME}/vps-run-outputs/{it}")
    

let fails =
    vpsRunOutputDirs
    |> Seq.map DirectoryInfo
    |> Seq.collect (fun it -> it.GetFiles())
    |> Seq.filter (fun it -> it.Name = "failed.txt")
    |> Seq.collect (fun it -> File.ReadAllLines(it.FullName))
    |> Set.ofSeq
    

let handlesToBeCopied =
    vpsRunOutputDirs
    |> Seq.map (fun it -> DirectoryInfo($"{it}/tweets"))
    |> Seq.collect (fun it -> it.GetFiles())
    |> Seq.map (fun it -> it.Name.Replace(".json", ""))
    |> Seq.filter (fun it -> not (fails.Contains(it)))
    |> Set.ofSeq


let copyBioFiles() =
    vpsRunOutputDirs
    |> Seq.map (fun it -> DirectoryInfo($"{it}/bios"))
    |> Seq.collect (fun it -> it.GetFiles())
    |> Seq.filter (fun it -> handlesToBeCopied.Contains(it.Name.Replace(".txt", "")))
    |> Seq.filter (fun it -> not (File.Exists($"{INTERIM}/{DATA_NAME}/bios/{it.Name}")))
    |> Seq.iter (fun it -> it.CopyTo($"{INTERIM}/{DATA_NAME}/bios/{it.Name}") |> ignore)
    

let copyTweetFiles() =
    vpsRunOutputDirs
    |> Seq.map (fun it -> DirectoryInfo($"{it}/tweets"))
    |> Seq.collect (fun it -> it.GetFiles())
    |> Seq.filter (fun it -> handlesToBeCopied.Contains(it.Name.Replace(".json", "")))
    |> Seq.filter (fun it -> not (File.Exists($"{INTERIM}/{DATA_NAME}/tweets/{it.Name}")))
    |> Seq.iter (fun it -> it.CopyTo($"{INTERIM}/{DATA_NAME}/tweets/{it.Name}") |> ignore)


[<EntryPoint>]
let main _ =
    copyBioFiles()
    copyTweetFiles()
    0 