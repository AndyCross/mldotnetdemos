using System;
using System.IO;
using System.Linq;
using Parquet;

namespace parquetdemo
{
    public class SimpleStructure
    {
        public int Id { get; set; }

        public string Name { get; set; }
        public DateTimeOffset FutureDate { get; set;}
        public double SomeNumeric { get; set; }
        public int? MaybeAnInt { get; set; }
    }

    class Program
    {        
        static readonly string _parquetPath = Path.Combine(Environment.CurrentDirectory, "data", "myParquetFile.parquet");

        static void Main(string[] args)
        {
            SimpleStructure[] structures = Enumerable
                .Range(0, 1000)
                .Select(i => new SimpleStructure
                {
                    Id = i,
                    Name = $"row {i}",
                    FutureDate = DateTime.UtcNow.AddDays(i),
                    SomeNumeric = 1D/i,
                    MaybeAnInt = (i % 44 == 0) ? (Nullable<int>)i : null
                })
                .ToArray();
            
            using (FileStream stream = File.Create(_parquetPath)) {
                ParquetConvert.Serialize(structures, stream);
            }

            using (FileStream stream = File.OpenRead(_parquetPath)) {
                structures = ParquetConvert.Deserialize<SimpleStructure>(stream);
                Console.WriteLine($"Roundtrip completed on {structures.Length} rows");
            }
        }
    }
}
