FROM mcr.microsoft.com/dotnet/runtime:5.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:5.0 AS build
WORKDIR /src
COPY ["Data.DatasetGeneration.SemiAutoLabel.HandLabel/Data.DatasetGeneration.SemiAutoLabel.HandLabel.fsproj", "Data.DatasetGeneration.SemiAutoLabel.HandLabel/"]
RUN dotnet restore "Data.DatasetGeneration.SemiAutoLabel.HandLabel/Data.DatasetGeneration.SemiAutoLabel.HandLabel.fsproj"
COPY . .
WORKDIR "/src/Data.DatasetGeneration.SemiAutoLabel.HandLabel"
RUN dotnet build "Data.DatasetGeneration.SemiAutoLabel.HandLabel.fsproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "Data.DatasetGeneration.SemiAutoLabel.HandLabel.fsproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "Data.DatasetGeneration.SemiAutoLabel.HandLabel.dll"]
