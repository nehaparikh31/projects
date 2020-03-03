package cps534.count;

import java.time.Duration;
import java.time.Instant;
import java.util.concurrent.TimeUnit;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class CountApplication extends Configured implements Tool 
{
	public static void main(String[] args) throws Exception 
	{	
		int res = ToolRunner.run(new Configuration(), new CountApplication(), args);
		System.exit(res);
	}

	public int run(String[] args) throws Exception 
	{
		//long t1 = System.currentTimeMillis();
		
		if (args.length != 2) 
		{
			System.out.println("usage: [input] [output]");
			//System.out.println("T1: "+t1);
			System.exit(-1);
		}
		
		Instant t1 = Instant.now();
		
		Job job = Job.getInstance(new Configuration());
		
		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);
		
		job.setMapperClass(CountMapper.class);
		job.setReducerClass(CountReducer.class);
		
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		
		FileInputFormat.setInputPaths(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));

		job.setJarByClass(CountApplication.class);
		job.submit();
		
		Instant t2 = Instant.now();
		long T = Duration.between(t1, t2).toMillis();
		
		System.out.print("Time taken is: "+ T + "Milliseconds\n");
			
		return 0;
	}
}
