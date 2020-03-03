package cps534.count_v2;

import java.io.IOException;
import java.util.HashMap;
import java.util.Set;
import java.util.StringTokenizer;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;

public class CountMapper extends Mapper<Object, Text, Text, IntWritable>
{	
	private final static IntWritable one = new IntWritable(1);
	private Text word = new Text();
	
	public void map(Object key, Text value, Context context) throws IOException, InterruptedException 
    {
        HashMap<String,Integer> hashMap = new HashMap<String, Integer>();
        StringTokenizer itr = new StringTokenizer(value.toString());

        while(itr.hasMoreElements())
        {
            String string = itr.nextToken();
            Integer count = hashMap.get(string);
            if(count == null)
            {
            	count = 0;
           	}
            count+=1;
            hashMap.put(string,count);
        }

        Set<String> keySet = hashMap.keySet();
        
        for (String str : keySet) 
        {
             word.set(str);
             one.set(hashMap.get(str));
             context.write(word,one);
        }
    }
}
